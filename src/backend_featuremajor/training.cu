// training.cu — batch-SOM training loop over the whole corpus.
//
// Each epoch: assign every sample to its best-matching neuron (launch_bmu), measure how
// good and how stable the assignment is, then move the codebook toward that assignment
// (launch_update_factorized) at the current neighbourhood radius σ, and finally shrink σ.
// Training stops when the assignment has settled (QE flat AND stable) or at max_epochs.

#include "sparsesom/training.hpp"
#include "sparsesom/device_dataset.hpp"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

// Clean error on a failed CUDA call instead of a null pointer → illegal memory access.
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

namespace sparsesom {

// Quantisation error: the mean distance from each sample to its winning neuron. Summed in
// double (n_samples can be large) then divided by the sample count.
static double quantisation_error(const FeatureMajorCodebook& s) {
    double sum = thrust::reduce(thrust::device,
                                s.d_bmu_dist, s.d_bmu_dist + s.n_samples,
                                0.0, thrust::plus<double>());
    return sum / (double)s.n_samples;
}

// Stability: the fraction of samples whose winning neuron is identical to last epoch's.
// For each sample, equal_to(bmu1, bmu_prev) yields 0/1; summing gives the unchanged count.
static double stability(const FeatureMajorCodebook& s) {
    size_t same = thrust::inner_product(thrust::device,
                                        s.d_bmu1, s.d_bmu1 + s.n_samples, s.d_bmu_prev,
                                        (size_t)0,
                                        thrust::plus<size_t>(),
                                        thrust::equal_to<uint32_t>());
    return (double)same / (double)s.n_samples;
}

double evaluate_qe(const DeviceDataset& dds, FeatureMajorCodebook& state, cudaStream_t stream) {
    launch_bmu(dds, state, stream);          // assign with the current (e.g. loaded) codebook
    cudaStreamSynchronize(stream);
    return quantisation_error(state);        // mean distance to BMU; no weight update
}

// Topographic error: 1 if a sample's bmu1 and bmu2 are not grid-adjacent (Chebyshev > 1).
static __global__ void te_kernel(const uint32_t* __restrict__ bmu1,
                                 const uint32_t* __restrict__ bmu2,
                                 uint32_t n, uint32_t cols, float* __restrict__ out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t c1 = bmu1[i], c2 = bmu2[i];
    int dr = (int)(c1 / cols) - (int)(c2 / cols); dr = dr < 0 ? -dr : dr;
    int dc = (int)(c1 % cols) - (int)(c2 % cols); dc = dc < 0 ? -dc : dc;
    out[i] = ((dr > dc ? dr : dc) > 1) ? 1.0f : 0.0f;     // Chebyshev (8-neighbour) distance > 1
}

double topographic_error(const FeatureMajorCodebook& s, cudaStream_t stream) {
    const uint32_t n = s.n_samples;
    if (n == 0) return 0.0;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));
    const uint32_t blk = 256, grd = (n + blk - 1) / blk;
    te_kernel<<<grd, blk, 0, stream>>>(s.d_bmu1, s.d_bmu2, n, s.cfg.map_cols, d_out);
    cudaStreamSynchronize(stream);
    double sum = thrust::reduce(thrust::device, d_out, d_out + n, 0.0, thrust::plus<double>());
    cudaFree(d_out);
    return sum / (double)n;
}

// Per-neuron win counter: each sample bumps its bmu1's slot.
static __global__ void hit_kernel(const uint32_t* __restrict__ bmu1, uint32_t n,
                                  uint32_t* __restrict__ counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[bmu1[i]], 1u);
}

double dead_fraction(const FeatureMajorCodebook& s, cudaStream_t stream) {
    const uint32_t K = s.n_neurons, n = s.n_samples;
    if (K == 0) return 0.0;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)K * sizeof(uint32_t)));
    cudaMemsetAsync(d_counts, 0, (size_t)K * sizeof(uint32_t), stream);
    if (n) { const uint32_t blk = 256, grd = (n + blk - 1) / blk;
             hit_kernel<<<grd, blk, 0, stream>>>(s.d_bmu1, n, d_counts); }
    cudaStreamSynchronize(stream);
    uint32_t dead = thrust::count(thrust::device, d_counts, d_counts + K, 0u);  // neurons with no wins
    cudaFree(d_counts);
    return (double)dead / (double)K;
}

// Per-sample cosine distance to BMU, recovered from the Euclidean BMU outputs:
//   dot = (‖x‖² + ‖w‖² − ‖x−w‖²) / 2,  cos_dist = 1 − dot/(‖x‖·‖w‖)
static __global__ void cosine_dist_kernel(
        const uint32_t* __restrict__ bmu1,
        const float*    __restrict__ bmu_dist,
        const float*    __restrict__ norms,
        const uint32_t* __restrict__ row_ptr,
        uint32_t n, float* __restrict__ out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xnorm2 = (float)(row_ptr[i + 1] - row_ptr[i]);   // binary: ‖x‖² = nnz
    float wnorm2 = norms[bmu1[i]];
    float dist2  = bmu_dist[i] * bmu_dist[i];             // d_bmu_dist stores ||x−w||, need ||x−w||²
    float dot    = (xnorm2 + wnorm2 - dist2) * 0.5f;
    float denom  = sqrtf(xnorm2) * sqrtf(wnorm2);
    out[i] = (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

double cosine_qe(const FeatureMajorCodebook& s, const uint32_t* d_row_ptr, cudaStream_t stream) {
    const uint32_t n = s.n_samples;
    if (n == 0) return 0.0;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));
    const uint32_t blk = 256, grd = (n + blk - 1) / blk;
    cosine_dist_kernel<<<grd, blk, 0, stream>>>(
        s.d_bmu1, s.d_bmu_dist, s.d_norms, d_row_ptr, n, d_out);
    cudaStreamSynchronize(stream);
    double sum = thrust::reduce(thrust::device, d_out, d_out + n, 0.0, thrust::plus<double>());
    cudaFree(d_out);
    return sum / (double)n;
}

double mean_bmu_dist(const FeatureMajorCodebook& s) {
    double sum = thrust::reduce(thrust::device,
                                s.d_bmu_dist, s.d_bmu_dist + s.n_samples,
                                0.0, thrust::plus<double>());
    return sum / (double)s.n_samples;
}

// σ (box half-width) at progress s for a given rate (the watchdog may raise rate above
// cfg.sigma_rate). s=0 → σ₀; s→∞ → σ_min; monotone non-increasing.
static float sigma_at(const SomConfig& cfg, float sigma0, double s, double rate) {
    double span = (double)sigma0 - (double)cfg.sigma_min;
    if (span <= 0.0) return cfg.sigma_min;
    double f = (cfg.sigma_schedule == SigmaSchedule::Linear)
             ? (1.0 - std::min(1.0, rate * s))     // linear ramp, clamped at the floor
             : std::exp(-rate * s);                // exponential / asymptotic (default)
    return (float)((double)cfg.sigma_min + span * f);
}

float sigma_for_progress(const SomConfig& cfg, float sigma0, double s) {
    return sigma_at(cfg, sigma0, s, (double)cfg.sigma_rate);
}

TrainResult train(const DeviceDataset& dds, FeatureMajorCodebook& state,
                  float sigma_start, uint32_t max_epochs, cudaStream_t stream) {
    const SomConfig& cfg = state.cfg;
    const bool  use_kl = (cfg.stop_criterion == StopCriterion::KaskiLagus);
    const char* crit   = use_kl ? "Kaski-Lagus" : "QE";

    const float    sigma0      = (sigma_start >= 0.0f) ? sigma_start : cfg.sigma_init;  // σ₀
    const uint32_t max_ep      = max_epochs ? max_epochs : cfg.max_epochs;             // epoch cap
    const double   sigma_small = std::max(1.0, 2.0 * (double)cfg.sigma_min);  // refinement regime
    const bool     det_sched   = (cfg.sigma_schedule == SigmaSchedule::DeterministicExp);
    double         rate        = cfg.sigma_rate;   // mutable: the watchdog may accelerate it (adaptive only)

    std::vector<EpochMetrics> history;
    history.reserve(max_ep);
    double     s = 0.0;                         // progress, advanced by windowed relative KL improvement
    double     prev_stop = 0.0;                 // previous epoch's stop-metric value
    uint32_t   flat_streak = 0;                 // consecutive flat epochs (plateau)
    StopReason reason = StopReason::MaxEpochs;
    float      sigma = sigma0;
    double     path_peak = 0.0;                 // max KL topology (path) term seen — for restart test
    double     bmu_s = 0.0, upd_s = 0.0;        // cumulative BMU- and update-phase wall (synced)
    using clk = std::chrono::steady_clock;

    const char* sched_name = det_sched ? "det-exp" :
                             (cfg.sigma_schedule == SigmaSchedule::Linear ? "linear" : "exp");
    std::printf("training: %u neurons, %u features, %u samples; "
                "sigma0=%.2f -> sigma_min=%.2f (%s, rate=%.3g), max %u epochs\n",
                state.n_neurons, state.n_features, state.n_samples,
                (double)sigma0, (double)cfg.sigma_min, sched_name,
                (double)rate, max_ep);
    std::printf("  stop: %s-plateau (rel<%.4g x %u) in refinement (sigma<=%.2f); watchdog: accel "
                "x%.3g, restart-if-path>%.2g*peak; KL window=%u, g_ref=%.3g\n",
                crit, (double)cfg.qe_rel_thresh, cfg.plateau_window, sigma_small,
                (double)cfg.sigma_accel, (double)cfg.wd_path_frac,
                cfg.sigma_window, (double)cfg.sigma_g_ref);

    for (uint32_t ep = 0; ep < max_ep; ++ep) {
        // Keep last epoch's assignment so stability() can see what changed.
        cudaMemcpyAsync(state.d_bmu_prev, state.d_bmu1,
                        (size_t)state.n_samples * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, stream);

        // Assign every sample to its best-matching neuron with the current codebook.
        auto t_bmu = clk::now();
        launch_bmu(dds, state, stream);
        cudaStreamSynchronize(stream);
        bmu_s += std::chrono::duration<double>(clk::now() - t_bmu).count();

        EpochMetrics m;
        m.epoch     = ep;
        m.qe        = quantisation_error(state);
        m.stability = stability(state);
        m.kl        = kaski_lagus(state, stream);    // always computed: it drives the σ schedule
        m.te        = topographic_error(state, stream);
        m.dead_frac = dead_fraction(state, stream);

        // Advance the progress variable that drives σ.
        if (det_sched) {
            // Deterministic: s = epoch index. σ is a pure function of the epoch count.
            s = (double)ep;
        } else if (ep >= 1) {
            // Adaptive: windowed relative KL improvement gates the progress —
            // s barely moves while KL is genuinely dropping (σ stays wide for ordering)
            // and advances toward 1/epoch once improvement flattens (refinement).
            uint32_t w = (ep >= cfg.sigma_window) ? cfg.sigma_window : ep;
            double ref_kl = history[ep - w].kl;
            double g_win = (ref_kl > 1e-12) ? std::max(0.0, (ref_kl - m.kl) / ref_kl) : 0.0;
            s += 1.0 - g_win / (g_win + (double)cfg.sigma_g_ref);
        }
        sigma   = sigma_at(cfg, sigma0, s, rate);    // endpoint interpolation (box half-width)
        m.sigma = sigma;
        history.push_back(m);

        double kl_path = m.kl - m.qe;                // Kaski–Lagus topology (path) term
        path_peak = std::max(path_peak, kl_path);    // track its peak for the restart detector
        std::printf("epoch %3u  sigma=%6.2f  QE=%.4f  KL=%.4f  path=%.4f  "
                    "stab=%.4f  TE=%.3f  dead=%.1f%%\n",
                    ep, (double)sigma, m.qe, m.kl, kl_path, m.stability, m.te, 100.0 * m.dead_frac);
        std::fflush(stdout);

        // Plateau of the stop metric (relative change below ε; absolute fallback when ~0).
        double stop_val = use_kl ? m.kl : m.qe;
        if (ep >= 1) {
            double rel = (prev_stop > 1e-12) ? std::fabs(stop_val - prev_stop) / prev_stop
                                             : (std::fabs(stop_val - prev_stop) < 1e-12 ? 0.0 : 1.0);
            if (rel < (double)cfg.qe_rel_thresh) ++flat_streak; else flat_streak = 0;
        }
        prev_stop = stop_val;

        // Watchdog + stop, acting asymmetrically on a plateau:
        if (flat_streak >= cfg.plateau_window) {
            if ((double)sigma <= sigma_small) {          // already in the refinement regime
                // Topology resolved requires BOTH: (a) the KL path (topology) term has fallen to
                // ≤ wd_path_frac of its peak (trajectory-relative), AND (b) the topographic error
                // is acceptable. The path term alone can read "resolved" on a collapsed/disordered
                // map (its inter-neuron weight distances shrink), so the direct TE check guards
                // against declaring convergence on a map that never grid-ordered (TE≈1).
                bool path_ok = (path_peak <= 1e-9) ||
                               (kl_path <= (double)cfg.wd_path_frac * path_peak);
                bool te_ok   = (m.te <= (double)cfg.te_converge_max);
                if (path_ok && te_ok) {
                    std::printf("converged after epoch %u (%s plateau; sigma=%.2f, "
                                "path=%.4f <= %.2g*peak, TE=%.3f <= %.2f)\n",
                                ep, crit, (double)sigma, kl_path, (double)cfg.wd_path_frac,
                                m.te, (double)cfg.te_converge_max);
                    reason = StopReason::Converged;
                } else {                                  // topology never ordered → restart slower
                    std::printf("RESTART_SLOWER after epoch %u: topology unresolved at small "
                                "sigma=%.2f (path=%.4f vs %.2g*peak=%.4f; TE=%.3f vs max %.2f)\n",
                                ep, (double)sigma, kl_path, (double)cfg.wd_path_frac, path_peak,
                                m.te, (double)cfg.te_converge_max);
                    reason = StopReason::RestartSlower;
                }
                break;
            }
            // Plateau while σ is still large → accelerate the decay to force refinement (safe:
            // shrinking σ never destroys existing order). Reset the streak to re-assess after.
            // Deterministic schedule: no rate mutation — σ follows the fixed exponential.
            if (!det_sched) {
                rate *= (double)cfg.sigma_accel;
                std::printf("  watchdog: plateau at large sigma=%.2f -> accelerate rate to %.3g\n",
                            (double)sigma, rate);
            }
            flat_streak = 0;
        }

        // Move the codebook toward the current assignment; refresh norms for the next BMU.
        auto t_upd = clk::now();
        launch_update_factorized(dds, state, sigma, stream);
        state.recompute_norms();                 // recompute_norms() syncs, so this wall is accurate
        upd_s += std::chrono::duration<double>(clk::now() - t_upd).count();
    }

    TrainResult result{ std::move(history), reason, reason == StopReason::Converged, sigma };
    result.bmu_seconds = bmu_s;
    result.update_seconds = upd_s;
    return result;
}

} // namespace sparsesom
