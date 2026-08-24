// test_kaski_lagus.cpp — verify the Kaski–Lagus distortion against a hand-computed value.
//
// 3×3 map, 2 features. Neuron k is given weight vector (k, 0), so the input-space distance
// between neurons p and q is exactly |p − q|. We set bmu1/bmu2/bmu_dist by hand and check the
// measure = mean over samples of (bmu_dist + sum of |Δ| along the 4-connected L-path bmu1→bmu2).

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/training.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace sparsesom;

int main() {
    SomConfig cfg;
    cfg.map_rows = 3; cfg.map_cols = 3;      // 9 neurons, grid index = r*3 + c
    cfg.n_features  = 2;
    cfg.kl_monitor_cap = 0;                  // all samples
    cfg.kl_path_radius = 8;
    const uint32_t K = cfg.n_neurons();      // 9
    const uint32_t V = cfg.n_features;          // 2
    const uint32_t A = 2;

    // Neuron k = (k, 0), feature-major: row v=0 is [0..8], row v=1 is all zeros.
    std::vector<float> hWfm((size_t)V * K, 0.0f);
    for (uint32_t k = 0; k < K; ++k) hWfm[0 * K + k] = (float)k;

    FeatureMajorCodebook cb(cfg, A);
    cb.upload_W_fm(hWfm);                    // float→half + recompute_norms; d_norms[k] = k^2

    // Sample 0: bmu1=0 (0,0), bmu2=4 (1,1), dist 0.5.  L-path 0→3→4: |0-3|+|3-4| = 3+1 = 4.
    // Sample 1: bmu1=2 (0,2), bmu2=6 (2,0), dist 1.0.  L-path 2→5→8→7→6: 3+3+1+1 = 8.
    std::vector<uint32_t> bmu1 = {0, 2}, bmu2 = {4, 6};
    std::vector<float>    dist = {0.5f, 1.0f};
    cudaMemcpy(cb.d_bmu1,     bmu1.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cb.d_bmu2,     bmu2.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cb.d_bmu_dist, dist.data(), A * sizeof(float),    cudaMemcpyHostToDevice);

    double got = kaski_lagus(cb);
    double expected = ((0.5 + 4.0) + (1.0 + 8.0)) / 2.0;   // = 6.75

    if (std::fabs(got - expected) > 1e-4) {
        std::printf("FAIL: kaski_lagus got %f, expected %f\n", got, expected);
        std::printf("kaski_lagus test FAILED\n");
        return 1;
    }
    std::printf("kaski_lagus test OK (KL=%f)\n", got);
    return 0;
}
