// pca_init.cu — initialize a feature-major FP16 codebook from PCA components.
//
// The .sompca file contains: mean[V], pc[k×V], sv[k] — roughly 360 KB for k=2, V=30766.
// A single kernel generates the full d_W_fm[V×K] as __half directly in GPU memory, with
// no host-side codebook buffer and no disk write.

#include "sparsesom/pca_init.hpp"
#include "sparsesom/feature_major_codebook.hpp"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace sparsesom {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

namespace {

constexpr char kMagic[8] = {'S', 'O', 'M', 'P', 'C', 'A', '1', '\0'};

struct SompcaHeader {
    char     magic[8];
    uint32_t n_features;
    uint32_t n_components;
    uint32_t n_samples;
    uint32_t reserved;
};

// Each thread computes one element of d_W_fm[v * K + k].
// Components are small (3 × V floats ≈ 370 KB) and cached in L1/L2; consecutive
// threads share the same v so reads are broadcast-efficient.
__global__ void pca_init_kernel(
    __half* __restrict__ W_fm,
    const float* __restrict__ d_mean,
    const float* __restrict__ d_pc0,
    const float* __restrict__ d_pc1,
    float scale0, float scale1,
    uint32_t K, uint32_t V,
    uint32_t map_rows, uint32_t map_cols
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)K * V;
    if (idx >= total) return;

    uint32_t k = (uint32_t)(idx % K);
    uint32_t v = (uint32_t)(idx / K);
    uint32_t r = k / map_cols;
    uint32_t c = k % map_cols;

    float r_norm = (map_rows > 1) ? (2.0f * r / (float)(map_rows - 1) - 1.0f) : 0.0f;
    float c_norm = (map_cols > 1) ? (2.0f * c / (float)(map_cols - 1) - 1.0f) : 0.0f;

    float w = d_mean[v] + scale0 * r_norm * d_pc0[v] + scale1 * c_norm * d_pc1[v];
    W_fm[(uint64_t)v * K + k] = __float2half(w);
}

} // namespace

void pca_init(FeatureMajorCodebook& cb, const std::string& sompca_path,
              uint32_t /*n_train_samples*/) {
    std::ifstream f(sompca_path, std::ios::binary);
    if (!f) throw std::runtime_error("pca_init: cannot open " + sompca_path);

    SompcaHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || std::memcmp(hdr.magic, kMagic, 8) != 0)
        throw std::runtime_error("pca_init: bad magic in " + sompca_path);
    if (hdr.n_features != cb.n_features)
        throw std::runtime_error("pca_init: .sompca has " + std::to_string(hdr.n_features) +
                                 " features but codebook has " + std::to_string(cb.n_features));
    if (hdr.n_components < 2)
        throw std::runtime_error("pca_init: need >= 2 components, got " +
                                 std::to_string(hdr.n_components));

    const uint32_t V  = hdr.n_features;
    const uint32_t nc = hdr.n_components;
    const uint32_t n_pca_samples = hdr.n_samples;

    std::vector<float> svs(nc);
    f.read(reinterpret_cast<char*>(svs.data()), nc * sizeof(float));
    std::vector<float> mean(V);
    f.read(reinterpret_cast<char*>(mean.data()), V * sizeof(float));
    std::vector<float> components(nc * V);
    f.read(reinterpret_cast<char*>(components.data()), nc * V * sizeof(float));
    if (!f) throw std::runtime_error("pca_init: truncated data in " + sompca_path);

    // Scale by sv/√n where n is the sample count used to compute the PCA (from the file header),
    // so the codebook span matches the data's spread regardless of whether training uses a subsample.
    float scale0 = svs[0] / std::sqrt((float)n_pca_samples);
    float scale1 = svs[1] / std::sqrt((float)n_pca_samples);

    std::printf("pca_init: V=%u, sv=[%.2f, %.2f], scale=[%.6f, %.6f], n_pca=%u\n",
                V, svs[0], svs[1], scale0, scale1, n_pca_samples);

    float *d_mean = nullptr, *d_pc0 = nullptr, *d_pc1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mean, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pc0,  V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pc1,  V * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_mean, mean.data(), V * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pc0,  components.data(),       V * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pc1,  components.data() + V,   V * sizeof(float), cudaMemcpyHostToDevice));

    const uint32_t K = cb.n_neurons;
    const uint64_t total = (uint64_t)K * V;
    const uint32_t blk = 256;
    const uint64_t grd = (total + blk - 1) / blk;

    // Grid dimension check: CUDA limit is 2^31-1 blocks.
    if (grd > (uint64_t)INT32_MAX)
        throw std::runtime_error("pca_init: grid too large (" + std::to_string(grd) +
                                 " blocks) — map is too big for a single kernel launch");

    pca_init_kernel<<<(uint32_t)grd, blk>>>(
        cb.d_W_fm, d_mean, d_pc0, d_pc1, scale0, scale1,
        K, V, cb.cfg.map_rows, cb.cfg.map_cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_mean);
    cudaFree(d_pc0);
    cudaFree(d_pc1);

    cb.recompute_norms();
    std::printf("pca_init: initialized %u x %u map (%u neurons) from PCA\n",
                cb.cfg.map_rows, cb.cfg.map_cols, K);
}

} // namespace sparsesom
