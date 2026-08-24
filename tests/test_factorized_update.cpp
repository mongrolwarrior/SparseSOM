// test_factorized_update.cpp — end-to-end check of the factorized feature-major update.
//
// Tiny hand-computable case: 2 samples on a 3×3 map (9 neurons), 4 features, both
// samples winning neuron 0. With sigma = 0 the box blur is the identity (radius 0), so
// the update reduces to the plain per-neuron mean of the samples that landed there:
//     neuron 0 weight[v] = (sum of feature v over its samples) / (sample count)
// Every other neuron won nothing (Den == 0) and must be left untouched.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace sparsesom;

int main() {
    // Map / problem dimensions.
    const uint32_t V = 4;        // feature count
    const uint32_t A = 2;        // sample count
    SomConfig cfg;
    cfg.map_rows   = 3;   // 3 × 3 = 9 neurons
    cfg.map_cols   = 3;
    cfg.n_features    = V;   // feature dimension comes from the config
    cfg.box_passes = 1;
    const uint32_t K = cfg.n_neurons();   // 9

    // CSR corpus: sample 0 = {feature 0, feature 1}; sample 1 = {feature 0, feature 2}.
    std::vector<uint32_t> row_ptr = {0, 2, 4};       // sample a -> col_idx[row_ptr[a]..row_ptr[a+1])
    std::vector<uint16_t> col_idx = {0, 1, 0, 2};    // the present features, concatenated
    DeviceDataset dds;
    dds.upload(row_ptr.data(), col_idx.data(), A, (uint32_t)col_idx.size());

    // Codebook, started at all-zero so untouched neurons are clearly 0.
    FeatureMajorCodebook cb(cfg, A);
    cudaMemset(cb.d_W_fm, 0, (size_t)V * K * sizeof(__half));

    // Both samples win neuron 0.
    std::vector<uint32_t> bmu1 = {0, 0};
    cudaMemcpy(cb.d_bmu1, bmu1.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // One factorized update with sigma = 0 (identity neighbourhood).
    launch_update_factorized(dds, cb, /*sigma=*/0.0f);

    // Pull the codebook back: W_fm[v*K + k].
    std::vector<float> W;
    cb.download_W_fm(W);

    // Expected weights at neuron 0: feature counts {2,1,1,0} / 2 samples.
    const float expect0[V] = {1.0f, 0.5f, 0.5f, 0.0f};
    int fails = 0;
    for (uint32_t v = 0; v < V; ++v) {
        float got = W[(size_t)v * K + 0];
        if (std::fabs(got - expect0[v]) > 1e-6f) {
            std::printf("FAIL neuron0 feature %u: got %f, expected %f\n", v, got, expect0[v]);
            ++fails;
        }
    }
    // Neuron 1 won nothing -> every feature must still be 0.
    for (uint32_t v = 0; v < V; ++v) {
        float got = W[(size_t)v * K + 1];
        if (std::fabs(got) > 1e-6f) {
            std::printf("FAIL neuron1 feature %u: got %f, expected 0 (untouched)\n", v, got);
            ++fails;
        }
    }

    if (fails) { std::printf("factorized update test FAILED (%d)\n", fails); return 1; }
    std::printf("factorized update test OK\n");
    return 0;
}
