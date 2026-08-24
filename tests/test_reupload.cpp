// test_reupload.cpp — restoring a saved codebook onto the GPU is faithful.
//
// Randomise a codebook, save it to .somw, load it into a fresh codebook via upload_W_fm, and
// confirm the two produce identical BMU assignments and distances on the same corpus — i.e.
// the disk round-trip (download -> save -> load -> upload, incl. norm rebuild) is exact.

#include "sparsesom/codebook_io.hpp"
#include "sparsesom/device_dataset.hpp"
#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/training.hpp"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace sparsesom;

int main() {
    const uint32_t V = 8;
    const uint16_t patterns[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    std::vector<uint32_t> rp = {0};
    std::vector<uint16_t> ci;
    for (uint32_t a = 0; a < 16; ++a) {
        const uint16_t* p = patterns[a % 4];
        ci.push_back(p[0]); ci.push_back(p[1]);
        rp.push_back((uint32_t)ci.size());
    }
    DeviceDataset dds;
    dds.upload(rp.data(), ci.data(), 16, (uint32_t)ci.size());

    SomConfig cfg;
    cfg.map_rows = 6; cfg.map_cols = 6;
    cfg.n_features  = V;
    const uint32_t A = 16, K = cfg.n_neurons();

    // Reference codebook.
    FeatureMajorCodebook cbA(cfg, A);
    cbA.randomise(7);
    std::vector<float> WA;
    cbA.download_W_fm(WA);

    // Disk round-trip into a fresh codebook.
    const std::string path = "/tmp/sparsesom_reupload.somw";
    save_codebook(path, cfg.map_rows, cfg.map_cols, cfg.n_features, WA.data());
    CodebookFile cf = load_codebook(path);
    FeatureMajorCodebook cbB(cfg, A);
    cbB.upload_W_fm(cf.W);

    // Assign the corpus with both and compare.
    launch_bmu(dds, cbA);
    launch_bmu(dds, cbB);
    cudaDeviceSynchronize();

    std::vector<uint32_t> bmuA(A), bmuB(A);
    std::vector<float>    dA(A), dB(A);
    cudaMemcpy(bmuA.data(), cbA.d_bmu1, A * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bmuB.data(), cbB.d_bmu1, A * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(dA.data(), cbA.d_bmu_dist, A * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB.data(), cbB.d_bmu_dist, A * sizeof(float), cudaMemcpyDeviceToHost);

    int fails = 0;
    if (cf.W.size() != WA.size() || cf.W != WA) { std::printf("FAIL: weights changed on round-trip\n"); ++fails; }
    (void)K;
    for (uint32_t a = 0; a < A; ++a) {
        if (bmuA[a] != bmuB[a]) { std::printf("FAIL: bmu1 differs at %u (%u vs %u)\n", a, bmuA[a], bmuB[a]); ++fails; }
        if (dA[a] != dB[a])     { std::printf("FAIL: bmu_dist differs at %u\n", a); ++fails; }
    }

    if (fails) { std::printf("reupload test FAILED (%d)\n", fails); return 1; }
    std::printf("reupload test OK (restored codebook reproduces identical assignments)\n");
    return 0;
}
