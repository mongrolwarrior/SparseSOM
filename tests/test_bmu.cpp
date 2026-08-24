// test_bmu.cpp — checks the BMU pass against a host reference.
//
// A 2×2 map (4 neurons) over 3 features, with hand-set codebook weights and 4 binary
// samples. For each sample we recompute the winning neuron (argmax 2·x·w − ‖w‖², i.e. the
// nearest neuron in Euclidean distance), its runner-up, and the distance on the host, and
// compare to the GPU result.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace sparsesom;

int main() {
    const uint32_t V = 3;   // features
    const uint32_t K = 4;   // neurons (2×2)
    const uint32_t A = 4;   // samples

    // Codebook in feature-major layout W_fm[v*K + k]. Neuron weight vectors (node-major):
    //   n0 = (.90,.15,.10)  n1 = (.20,.85,.25)  n2 = (.12,.30,.88)  n3 = (.55,.50,.45)
    // Deliberately asymmetric so there are no score ties.
    std::vector<float> hWfm = {
        0.90f, 0.20f, 0.12f, 0.55f,   // feature 0 across the 4 neurons
        0.15f, 0.85f, 0.30f, 0.50f,   // feature 1
        0.10f, 0.25f, 0.88f, 0.45f,   // feature 2
    };

    // Samples: {0}, {1}, {2}, {0,1,2}.
    std::vector<uint32_t> row_ptr = {0, 1, 2, 3, 6};
    std::vector<uint16_t> col_idx = {0, 1, 2, 0, 1, 2};
    DeviceDataset dds;
    dds.upload(row_ptr.data(), col_idx.data(), A, (uint32_t)col_idx.size());

    auto wfm = [&](uint32_t v, uint32_t k) { return (double)hWfm[(size_t)v * K + k]; };

    int fails = 0;
    SomConfig cfg;
    cfg.map_rows = 2;
    cfg.map_cols = 2;
    cfg.n_features = V;

    FeatureMajorCodebook cb(cfg, A);
    cb.upload_W_fm(hWfm);                 // float→half + recompute_norms

    launch_bmu(dds, cb);

    std::vector<uint32_t> bmu1(A), bmu2(A);
    std::vector<float>    dist(A);
    cudaMemcpy(bmu1.data(), cb.d_bmu1,     A * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bmu2.data(), cb.d_bmu2,     A * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(dist.data(), cb.d_bmu_dist, A * sizeof(float),    cudaMemcpyDeviceToHost);

    for (uint32_t a = 0; a < A; ++a) {
        // Host reference: top-2 Euclidean scores (2·x·w − ‖w‖²), then distance to the best.
        uint32_t feat_lo = row_ptr[a], feat_hi = row_ptr[a + 1];
        double   best_score = -1e300, second_score = -1e300, best_dot = 0;
        uint32_t best_k = 0, second_k = 0;
        for (uint32_t k = 0; k < K; ++k) {
            double nrm = 0;                                     // ‖w_k‖²
            for (uint32_t v = 0; v < V; ++v) nrm += wfm(v, k) * wfm(v, k);
            double dot = 0;                                     // x·w over the sample's features
            for (uint32_t j = feat_lo; j < feat_hi; ++j) dot += wfm(col_idx[j], k);
            double score = 2.0 * dot - nrm;
            if (score > best_score) {
                second_score = best_score; second_k = best_k;
                best_score = score; best_k = k; best_dot = dot;
            } else if (score > second_score) {
                second_score = score; second_k = k;
            }
        }
        double nrm_best = 0;                                    // ‖w‖² of the winner
        for (uint32_t v = 0; v < V; ++v) nrm_best += wfm(v, best_k) * wfm(v, best_k);
        double nnz = (double)(feat_hi - feat_lo);
        double exp_dist = std::sqrt(std::fmax(0.0, nnz - 2.0 * best_dot + nrm_best));

        if (bmu1[a] != best_k) {
            std::printf("FAIL sample %u: bmu1 got %u, expected %u\n", a, bmu1[a], best_k);
            ++fails;
        }
        if (bmu2[a] != second_k) {
            std::printf("FAIL sample %u: bmu2 got %u, expected %u\n", a, bmu2[a], second_k);
            ++fails;
        }
        if (std::fabs(dist[a] - exp_dist) > 1e-3) {
            std::printf("FAIL sample %u: dist got %f, expected %f\n", a, dist[a], exp_dist);
            ++fails;
        }
    }

    if (fails) { std::printf("BMU test FAILED (%d)\n", fails); return 1; }
    std::printf("BMU test OK\n");
    return 0;
}
