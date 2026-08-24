#pragma once
// training.hpp — the batch-SOM training loop that drives BMU assignment and the
// factorized weight update over epochs with a decaying neighbourhood. Include from .cu
// only (CUDA types).

#include "sparsesom/feature_major_codebook.hpp"
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace sparsesom {

struct DeviceDataset;  // device_dataset.hpp

// One row of training history, recorded per epoch.
struct EpochMetrics {
    uint32_t epoch;       // 0-based epoch index
    float    sigma;       // neighbourhood radius used this epoch
    double   qe;          // quantisation error: mean distance from each sample to its BMU
    double   stability;   // fraction of samples whose BMU is unchanged from the previous epoch
    double   kl;          // Kaski–Lagus distortion (mean QE + BMU→2nd-BMU grid path); 0 if not computed
    // ── logged diagnostics (reported, never a stop trigger) ──
    double   te;          // topographic error: fraction of samples whose bmu1/bmu2 aren't grid-adjacent
    double   dead_frac;   // fraction of neurons that won no sample this epoch
};

// Why a train() call stopped.
//   Converged    — the stop metric plateaued in the refinement regime (σ small, topology ok).
//   MaxEpochs    — hit the per-call epoch cap without converging.
//   RestartSlower — watchdog flag: σ reached small but topology was still poor; the caller
//                   should restart this run with a slower σ rate (no in-flight repair).
enum class StopReason { Converged, MaxEpochs, RestartSlower };

// Outcome of a train() call.
struct TrainResult {
    std::vector<EpochMetrics> history;     // per-epoch metrics for this call (last row = final state)
    StopReason stop_reason = StopReason::MaxEpochs;
    bool  converged = false;               // == (stop_reason == Converged), for convenience
    float sigma_end = 0.0f;                 // σ after the last epoch — pass back as sigma_start to continue
    double bmu_seconds = 0.0;               // cumulative wall in the BMU/assignment phase
    double update_seconds = 0.0;            // cumulative wall in the codebook-update phase
};

// The scheduled σ (box half-width) at progress s, given σ₀ and cfg's schedule type/rate.
// Pure helper (exposed for testing): s=0 → σ₀; s→∞ → σ_min; monotone non-increasing.
float sigma_for_progress(const SomConfig& cfg, float sigma0, double s);

// Train the SOM on the whole corpus until the stop criterion plateaus or the epoch cap is hit.
// Each epoch: assign every sample to its BMU, measure the metrics, move the codebook toward the
// assignment with the factorized update at the current σ, then decay σ. The codebook should be
// randomise()d (or restored) before the first call.
//   sigma_start < 0  → start a fresh schedule at cfg.sigma_init.
//   sigma_start >= 0 → CONTINUE from that σ (the schedule carries over rather than re-broadening);
//                      pass a previous result's sigma_end to train for more epochs.
//   max_epochs == 0  → use cfg.max_epochs; otherwise cap this call to max_epochs epochs.
TrainResult train(const DeviceDataset& dds, FeatureMajorCodebook& state,
                  float sigma_start = -1.0f, uint32_t max_epochs = 0,
                  cudaStream_t stream = nullptr);

// Assign every sample to its BMU with the current codebook and return the quantisation error
// (mean distance to BMU). Does NOT modify the codebook — use it to apply an already-trained
// map (e.g. one restored with upload_W_fm) to a corpus.
double evaluate_qe(const DeviceDataset& dds, FeatureMajorCodebook& state,
                   cudaStream_t stream = nullptr);

// Mean Kaski–Lagus distortion over the monitored samples (see kaski_lagus.cu). Reads the
// codebook's current d_bmu1/d_bmu2/d_bmu_dist (run launch_bmu first) and cfg.kl_* settings.
double kaski_lagus(const FeatureMajorCodebook& state, cudaStream_t stream = nullptr);

// Diagnostics (run launch_bmu first). Reported by train(), never used as a stop trigger.
//   topographic_error — fraction of samples whose bmu1 and bmu2 are not grid-adjacent
//                       (Chebyshev / 8-neighbour distance > 1): a topology-violation rate.
//   dead_fraction     — fraction of neurons that won no sample (bmu1) this pass.
double topographic_error(const FeatureMajorCodebook& state, cudaStream_t stream = nullptr);
double dead_fraction(const FeatureMajorCodebook& state, cudaStream_t stream = nullptr);

// Cosine QE: mean cosine distance (1 − cos_sim) from each sample to its BMU. Computed from
// the existing BMU outputs WITHOUT modifying the kernel:
//   dot = (‖x‖² + ‖w‖² − ‖x−w‖²) / 2      (from the identity ‖x−w‖² = ‖x‖²−2·dot+‖w‖²)
//   cos_sim = dot / (‖x‖ · ‖w‖)
// For binary samples ‖x‖² = nnz (the feature count). Requires d_bmu1, d_bmu_dist, and
// d_norms to be current (run launch_bmu first). d_row_ptr provides per-sample nnz.
double cosine_qe(const FeatureMajorCodebook& state, const uint32_t* d_row_ptr,
                 cudaStream_t stream = nullptr);

// Mean Euclidean BMU distance: mean of d_bmu_dist (‖x−w_bmu‖²) over all samples.
// Requires d_bmu_dist to be current (run launch_bmu first).
double mean_bmu_dist(const FeatureMajorCodebook& state);

} // namespace sparsesom
