#pragma once
// config.hpp — hyperparameters for a generic sparse-binary SOM. Nothing here is tied to
// any particular dataset: a SomConfig fully describes the map geometry, the input feature
// dimension, and the training knobs the feature-major backend reads. The map is always a
// square lattice with planar (clamped) boundaries.
//
// The default values reproduce the original MEDLINE "headline" configuration SparseSOM
// was built around (a 350×350 map over a ~30k-term feature space). They are only defaults —
// override any field for a different corpus.

#include <cstdint>

namespace sparsesom {

// Defaults for the MEDLINE headline configuration. Named here (rather than inlined as
// magic numbers) so the dataset-specific origin of each value is explicit.
namespace defaults {
inline constexpr uint32_t map_dim    = 350;     // square map edge → 350×350 = 122500 neurons
inline constexpr uint32_t n_features = 30000;   // input feature dimension
inline constexpr int      box_passes = 3;       // ≈ Gaussian neighbourhood
inline constexpr float    te_converge_max = 0.5f;  // max TE allowed to call a map "converged"
} // namespace defaults

// Which metric drives early-stopping.
//   KaskiLagus        — combined QE + topology distortion (single continuous input-space
//                       distance); the default. Uses bmu1 and bmu2.
//   QuantizationError — QE-plateau (needs only bmu1); the cheaper selectable alternative.
enum class StopCriterion { KaskiLagus, QuantizationError };

// How σ interpolates from σ₀ down to σ_min.
//   Exponential        — σ = σ_min + (σ₀−σ_min)·exp(−rate·s) where s is KL-improvement-driven;
//                         asymptotic, degrades gracefully when epoch count is unknown (the default).
//   Linear             — σ = σ₀ − (σ₀−σ_min)·min(1, rate·s); clamped at σ_min.
//   DeterministicExp   — same formula as Exponential but s = epoch index (no KL feedback, no
//                         watchdog acceleration). Reproduces the manuscript's fixed schedule.
enum class SigmaSchedule { Exponential, Linear, DeterministicExp };

// Which BMU kernel variant to use. All produce identical results (same argmin); only the
// memory access pattern and launch configuration differ.
enum class BmuKernel {
    Tile16,     // baseline: 16 samples/tile, 256 threads (current default)
    Tile32,     // 32 samples/tile, 256 threads — more codebook reuse
    Cluster,    // [INCOMPLETE] Tile32 + SM 9.0 thread block clusters — not yet benchmarked
    Tma,        // [INCOMPLETE] Tile32 + L2 prefetch — not yet benchmarked
};

struct SomConfig {
    // Map geometry (square lattice). n_neurons() == map_rows * map_cols.
    uint32_t map_rows = defaults::map_dim;   // number of neuron rows on the map
    uint32_t map_cols = defaults::map_dim;   // number of neuron columns on the map

    // Input feature dimension — how many distinct binary features an input sample can
    // have (the codebook is n_neurons × n_features). Set this to the dataset's feature count
    // before building the codebook; the default is the MEDLINE feature count.
    uint32_t n_features = defaults::n_features;

    // Factorized-update neighbourhood: number of separable box-blur passes per axis.
    // 1 = square "bubble"; 3 ≈ Gaussian (quality default, σ-independent). Both the
    // numerator and denominator get the same passes, so the box scale cancels in W=N/D.
    int box_passes = defaults::box_passes;

    // Topology gate for early-stopping: a map is only reported "converged" if its topographic
    // error is at most this. The KL path term alone can read "resolved" on a collapsed/disordered
    // map (its inter-neuron weight distances shrink), so a direct grid-topology check guards
    // against declaring convergence on a map that never ordered (TE≈1).
    float te_converge_max = defaults::te_converge_max;

    // ── Neighbourhood (σ) schedule — endpoint-interpolated, driven by KL improvement ──
    // σ is the box-filter HALF-WIDTH (radius) in grid cells for the factorized update's
    // separable box blur (box_passes passes ≈ a Gaussian; effective Gaussian width ≈ σ·√(P/3),
    // so at P=3 σ_box ≈ σ_gaussian). σ is NOT used by the BMU.
    //
    // Decoupled from epoch count (training length is set by the KL stop, not a fixed T): σ
    // interpolates from σ₀ to σ_min over a progress variable s advanced by the *relative
    // improvement* in the Kaski–Lagus metric — σ stays wide while KL drops fast (global
    // ordering), contracting toward σ_min as improvement flattens (refinement).
    //   σ₀ recommended = N/2 (half the map edge) so it scales across a size sweep — set by caller.
    //   σ_min ≈ 0.5 (sub-neuron). NOTE: the box radius is round(σ), so σ_min < 0.5 collapses fully
    //   to the BMU (radius 0); σ_min = 0.5 leaves a radius-1 window.
    float         sigma_init     = 100.0f;  // σ₀ — initial width (caller sets ≈ N/2)
    float         sigma_min      = 0.5f;    // σ floor (box half-width units)
    SigmaSchedule sigma_schedule = SigmaSchedule::Exponential;  // exp (default) | linear
    float         sigma_rate     = 0.3f;    // interpolation rate over the progress variable s
    // Fractional-KL-improvement scale that advances s. Progress is gated on a *windowed* net
    // improvement (over sigma_window epochs), not single-epoch dips, so transient/non-monotonic
    // KL bumps don't prematurely contract σ. g_ref is a PRIMARY sweep parameter (the harness
    // varies it); this default is only a starting point.
    float         sigma_g_ref    = 0.01f;
    uint32_t      sigma_window    = 3;      // epochs over which KL improvement is measured for s
    uint32_t      max_epochs     = 200;     // hard ceiling on epochs

    // ── Convergence + watchdog ───────────────────────────────────────────────────
    // Stop when the chosen metric's relative change stays < qe_rel_thresh for plateau_window
    // epochs AND σ has contracted to the refinement regime (σ ≤ ~2·σ_min). Watchdog, asymmetric:
    //   - plateau while σ still large → accelerate (rate ×= sigma_accel) to force refinement
    //     (safe: shrinking σ never destroys order);
    //   - σ already small but topology UNresolved → flag the run for restart with a slower rate
    //     (do NOT re-expand σ in-flight — that would discard ordering). "Unresolved" is judged
    //     trajectory-relatively: the KL path (topology) term must have fallen to ≤ wd_path_frac
    //     of its peak; if it's still above that, topology never ordered → restart slower.
    StopCriterion stop_criterion = StopCriterion::KaskiLagus;
    float         qe_rel_thresh  = 0.001f;  // plateau ε (relative per-epoch change counted "flat")
    uint32_t      plateau_window = 3;       // consecutive flat epochs to declare a plateau
    float         sigma_accel    = 2.0f;    // watchdog: rate multiplier on plateau-while-σ-large
    float         wd_path_frac   = 0.25f;   // restart unless KL path term fell to ≤ this × its peak

    // ── Kaski–Lagus distortion (the default stop metric) ─────────────────────────
    // Per sample: distance to its BMU + the grid-path length from BMU to 2nd-BMU, summed as
    // input-space edge distances along a 4-connected L-path, capped at kl_path_radius edges.
    uint32_t kl_path_radius  = 8;       // max path edges per sample (BMUs are almost always near)
    uint32_t kl_monitor_cap  = 100000;  // compute KL over the first min(cap, n_samples) samples; 0 = all
    // The path edge distance uses the PLAIN Euclidean norm by default — commensurable with the
    // QE term (which is itself plain Euclidean). The β-weighted-norm variant is a genuine design
    // choice, surfaced here rather than hardcoded; it is not implemented yet (setting true throws).
    bool     kl_path_score_weighted = false;

    // BMU kernel variant (all produce identical assignments; only performance differs).
    BmuKernel bmu_kernel = BmuKernel::Tile16;

    // Total neuron count on the map.
    uint32_t n_neurons() const noexcept { return map_rows * map_cols; }
};

} // namespace sparsesom
