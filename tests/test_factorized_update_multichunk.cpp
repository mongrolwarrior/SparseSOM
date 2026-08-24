// test_factorized_update_multichunk.cpp — exercises the feature-streaming path across
// more than one chunk, including a partial final chunk, on a non-zero neuron.
//
// The update streams features in chunks of CH = min(2048, V). With V = 2050 there are two
// chunks: a full [0,2048) and a partial [2048,2050). This is the case the earlier MedSOM
// code mishandled: the accumulate used the *valid* chunk width (2) as the tile stride
// while the blur/write used the full stride (2048), so features in the last chunk landed
// at the wrong neuron for any neuron index > 0. Here both samples win neuron 5, with
// features in BOTH chunks, so a stride mix-up would corrupt the result.
//
// sigma = 0 => identity blur => neuron 5 weight[v] = (count of feature v) / (sample count).

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace sparsesom;

int main() {
    const uint32_t V   = 2050;            // > 2048 -> two feature chunks (last one partial)
    const uint32_t A   = 2;               // sample count
    const uint32_t WIN = 5;               // the (non-zero) neuron both samples win
    SomConfig cfg;
    cfg.map_rows   = 3;   // 3 × 3 = 9 neurons
    cfg.map_cols   = 3;
    cfg.n_features    = V;   // feature dimension comes from the config
    cfg.box_passes = 1;
    const uint32_t K   = cfg.n_neurons(); // 9

    // sample 0 = {feature 0, feature 2048}; sample 1 = {feature 0, feature 2049}.
    // Feature 0 is in the first chunk; 2048/2049 are in the partial last chunk.
    std::vector<uint32_t> row_ptr = {0, 2, 4};
    std::vector<uint16_t> col_idx = {0, 2048, 0, 2049};
    DeviceDataset dds;
    dds.upload(row_ptr.data(), col_idx.data(), A, (uint32_t)col_idx.size());

    FeatureMajorCodebook cb(cfg, A);
    cudaMemset(cb.d_W_fm, 0, (size_t)V * K * sizeof(__half));

    std::vector<uint32_t> bmu1 = {WIN, WIN};
    cudaMemcpy(cb.d_bmu1, bmu1.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);

    launch_update_factorized(dds, cb, /*sigma=*/0.0f);

    std::vector<float> W;
    cb.download_W_fm(W);

    auto at = [&](uint32_t v, uint32_t k) { return W[(size_t)v * K + k]; };

    int fails = 0;
    auto check = [&](const char* what, float got, float exp) {
        if (std::fabs(got - exp) > 1e-6f) {
            std::printf("FAIL %s: got %f, expected %f\n", what, got, exp);
            ++fails;
        }
    };

    // Neuron 5: feature 0 in both samples -> 2/2 = 1.0; features 2048 & 2049 once each -> 0.5.
    check("neuron5 feature0",    at(0, WIN),    1.0f);
    check("neuron5 feature2048", at(2048, WIN), 0.5f);
    check("neuron5 feature2049", at(2049, WIN), 0.5f);

    // Neuron 0 won nothing: its last-chunk features must stay 0 (corrupted under the old bug).
    check("neuron0 feature2048", at(2048, 0), 0.0f);
    check("neuron0 feature2049", at(2049, 0), 0.0f);
    // And a first-chunk feature on the untouched neuron 5's neighbour stays 0 too.
    check("neuron4 feature0",    at(0, 4),     0.0f);

    if (fails) { std::printf("multi-chunk update test FAILED (%d)\n", fails); return 1; }
    std::printf("multi-chunk update test OK\n");
    return 0;
}
