#pragma once
// feature_major_codebook.hpp — GPU feature-major SOM codebook + the per-sample BMU
// arrays the factorized update consumes. Include from .cu only (CUDA types).
//
// The codebook is stored ONLY in feature-major layout:
//     d_W_fm[v * K + k]   (feature v, neuron k; K = n_neurons)
// i.e. for each feature the K neuron weights are contiguous. There is no node-major
// codebook — the factorized update writes d_W_fm directly via a tiled transpose, which
// is what coalesces the codebook write. Weights are FP32 (per-neuron feature means);
// the samples that drive them are sparse-binary (see DeviceDataset).

#include "sparsesom/config.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace sparsesom {

struct DeviceDataset;  // device_dataset.hpp

struct FeatureMajorCodebook {
    const SomConfig& cfg;

    uint32_t n_samples = 0;
    uint32_t n_features    = 0;   // V — feature count
    uint32_t n_neurons  = 0;   // K — map_rows * map_cols

    // Feature-major codebook [V × K]: d_W_fm[v * K + k]. Stored as FP16 to halve
    // bandwidth in the BMU kernel; all accumulation is done in FP32.
    __half* d_W_fm = nullptr;

    // Per-neuron squared L2 norm d_norms[k] = Σ_v w_vk^2 (rebuilt by recompute_norms());
    // used both for the Euclidean BMU score (2·x·w − ‖w‖²) and the reported distance.
    float* d_norms = nullptr;

    // Per-sample best-match unit [n_samples]: the index of the neuron that won each
    // sample in the most recent BMU pass. The factorized update reads it.
    uint32_t* d_bmu1 = nullptr;
    // Per-sample second-best unit [n_samples]: the runner-up neuron. Needed by topology
    // measures (Kaski–Lagus path term, topographic error).
    uint32_t* d_bmu2 = nullptr;
    // Previous epoch's d_bmu1 [n_samples], kept by the training loop so it can measure
    // stability (the fraction of samples whose winning neuron didn't change).
    uint32_t* d_bmu_prev = nullptr;
    // Per-sample Euclidean distance to that winning neuron [n_samples]. Filled by the
    // BMU pass; the basis for quantization-error reporting.
    float*    d_bmu_dist = nullptr;

    // n_samples = number of input samples (dataset size). The feature dimension and map
    // geometry come from the config (c.n_features, c.map_rows/cols).
    FeatureMajorCodebook(const SomConfig& c, uint32_t n_samples);
    ~FeatureMajorCodebook();
    FeatureMajorCodebook(const FeatureMajorCodebook&)            = delete;
    FeatureMajorCodebook& operator=(const FeatureMajorCodebook&) = delete;

    // Randomise the codebook in (0, 1/sqrt(V)) and rebuild norms.
    void randomise(unsigned seed = 42);

    // Recompute d_norms (squared L2) from d_W_fm.
    void recompute_norms();

    // Copy the feature-major codebook d_W_fm to host memory (resizes out to n_features*n_neurons,
    // feature-major W[v*K + k]). Used to persist a trained map via save_codebook().
    void download_W_fm(std::vector<float>& out) const;

    // Copy the per-sample winning-neuron indices d_bmu1 to host (resizes out to n_samples). After
    // an assignment pass this is the SOM clustering — point p's neuron — for recovery scoring.
    void download_bmu1(std::vector<uint32_t>& out) const;

    // Load a feature-major codebook (W[v*K + k], length n_features*n_neurons) from host memory
    // into d_W_fm and rebuild the norms, so the map is immediately ready for BMU/training.
    // Throws if W's length doesn't match this codebook's geometry. The inverse of
    // download_W_fm(); pair with load_codebook() to restore a saved .somw onto the GPU.
    void upload_W_fm(const std::vector<float>& W);

    // Stream the FP16 codebook to a .somw file in feature-chunks, converting to FP32 on the
    // fly. Uses ~K*chunk*4 GPU temp instead of the full V*K*4 that download_W_fm() needs.
    void save_codebook_chunked(const std::string& path, uint32_t chunk_features = 2048) const;

    // Reallocate the per-sample BMU arrays for a different sample count. The codebook
    // weights (d_W_fm, d_norms) are unchanged — use this to switch from the training
    // corpus to a held-out eval corpus without duplicating the codebook.
    void resize_samples(uint32_t new_n_samples);
};

// Factorized batch-SOM update (feature-streamed), writing the feature-major codebook.
// Accumulate per-BMU-node feature sums (atomic scatter, node-major tiles) → σ-independent
// separable box-blur convolution over the square map → W = Num/Den, transposed into
// d_W_fm. Processes the whole consolidated corpus (all n_samples). sigma sets the box
// radius (~the neighbourhood width).
void launch_update_factorized(const DeviceDataset& dds, FeatureMajorCodebook& state,
                              float sigma, cudaStream_t stream = nullptr);

// BMU (best-match-unit) pass: for every input sample, find the neuron that maximises the
// precision-weighted score against the feature-major codebook, writing d_bmu1 and the
// Euclidean d_bmu_dist. This is the "assignment" half of a training step; the factorized
// update is the "weight" half. Processes the whole corpus. Tile-based sparse-dense scoring
// (extracted from MedSOM-Naive's SpmmTile): the feature-major layout lets a warp read
// W_fm[v*K + k] coalesced and reuse it across the tile's samples that share feature v.
void launch_bmu(const DeviceDataset& dds, FeatureMajorCodebook& state,
                cudaStream_t stream = nullptr);

} // namespace sparsesom
