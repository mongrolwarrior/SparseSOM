// feature_major_codebook.cu — allocation, randomisation, and norm recompute for the
// feature-major codebook. Extracted from MedSOM-Naive's som_state.cu (fm-only path):
// the codebook lives only in feature-major layout, so randomise() draws straight into
// d_W_fm and recompute_norms() reads it column-strided.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/codebook_io.hpp"
#include <curand.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_fp16.h>

namespace sparsesom {

// Wrap a CUDA call: if it returns anything other than success, throw with the call text.
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

FeatureMajorCodebook::FeatureMajorCodebook(const SomConfig& c, uint32_t n_samples)
    : cfg(c), n_samples(n_samples), n_features(c.n_features), n_neurons(c.n_neurons()) {
    const uint32_t K = n_neurons;   // K = number of neurons (map cells) on the map

    // Codebook: K neurons × n_features, one FP16 weight each, feature-major.
    CUDA_CHECK(cudaMalloc(&d_W_fm,        (size_t)K * n_features * sizeof(__half)));
    // d_norms: one squared-L2 value per neuron, for the Euclidean BMU score and distance.
    CUDA_CHECK(cudaMalloc(&d_norms,       (size_t)K         * sizeof(float)));
    // d_bmu1 / d_bmu2 / d_bmu_prev / d_bmu_dist: per sample — the winning neuron, the
    // runner-up, the previous epoch's winner, and the distance to the current winner.
    CUDA_CHECK(cudaMalloc(&d_bmu1,        (size_t)n_samples     * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu2,        (size_t)n_samples     * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu_prev,    (size_t)n_samples     * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu_dist,    (size_t)n_samples     * sizeof(float)));

    // 0xFFFFFFFF in every byte = "no BMU assigned yet" sentinel for each sample.
    CUDA_CHECK(cudaMemset(d_bmu1,     0xFF, (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bmu2,     0xFF, (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bmu_prev, 0xFF, (size_t)n_samples * sizeof(uint32_t)));
}

FeatureMajorCodebook::~FeatureMajorCodebook() {
    cudaFree(d_W_fm);
    cudaFree(d_norms);
    cudaFree(d_bmu1);
    cudaFree(d_bmu2);
    cudaFree(d_bmu_prev);
    cudaFree(d_bmu_dist);
}

// Scale float src[i] by s and store as __half into dst[i].
static __global__ void scale_f2h_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                        uint64_t n, float s) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i] * s);
}

// Convert __half src to float dst, element-wise.
static __global__ void h2f_kernel(const __half* __restrict__ src, float* __restrict__ dst,
                                  uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// Convert float src to __half dst, element-wise.
static __global__ void f2h_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                  uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void FeatureMajorCodebook::randomise(unsigned seed) {
    const uint32_t K     = n_neurons;
    const uint64_t total = (uint64_t)K * n_features;
    const float    scale = 1.0f / std::sqrt(static_cast<float>(n_features));
    const uint32_t blk   = 256;

    // curand generates float; convert to __half in chunks to avoid allocating a full FP32 copy
    // of the codebook (which at large map sizes would exceed GPU memory).
    const uint64_t chunk = std::min(total, (uint64_t)K * std::min(n_features, 2048u));
    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, chunk * sizeof(float)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    for (uint64_t off = 0; off < total; off += chunk) {
        uint64_t n = std::min(chunk, total - off);
        curandGenerateUniform(gen, d_tmp, n);
        uint32_t grd = static_cast<uint32_t>((n + blk - 1) / blk);
        scale_f2h_kernel<<<grd, blk>>>(d_tmp, d_W_fm + off, n, scale);
    }
    curandDestroyGenerator(gen);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_tmp);

    recompute_norms();
}

// Compute, for each neuron k, the sum over all features v of W_fm[v][k] raised to `power`.
// One thread per neuron; consecutive threads handle consecutive k, so reading
// W_fm[v*K + k] for a fixed v is a coalesced (contiguous) memory access.
//   Wfm   = the feature-major codebook [V × K]
//   norms = output, one accumulated value per neuron
//   K     = neuron count (also the stride between successive features in Wfm)
//   V     = feature count (number of terms summed per neuron)
//   power = exponent applied to each weight (2 => squared L2 norm)
static __global__ void norm_fm_kernel(const __half* __restrict__ Wfm,
                                      float* __restrict__ norms,
                                      uint32_t K, uint32_t V, float power) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float acc = 0.0f;
    for (uint32_t v = 0; v < V; ++v) {
        float w = __half2float(Wfm[(size_t)v * K + k]);
        acc += (power == 2.0f) ? w * w : __powf(w, power);
    }
    norms[k] = acc;
}

void FeatureMajorCodebook::download_W_fm(std::vector<float>& out) const {
    const uint64_t total = (uint64_t)n_features * n_neurons;
    out.resize(total);
    // Convert __half → float on the GPU, then copy the float buffer to host.
    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));
    const uint32_t blk = 256;
    const uint32_t grd = static_cast<uint32_t>((total + blk - 1) / blk);
    h2f_kernel<<<grd, blk>>>(d_W_fm, d_tmp, total);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), d_tmp, total * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_tmp);
}

void FeatureMajorCodebook::save_codebook_chunked(const std::string& path,
                                                  uint32_t chunk_features) const {
    const uint32_t K = n_neurons;
    const uint32_t V = n_features;
    const uint32_t ch = std::min(chunk_features, V);
    const uint64_t chunk_elems = (uint64_t)K * ch;

    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, chunk_elems * sizeof(float)));
    std::vector<float> h_buf(chunk_elems);

    // Write the .somw header (same format as save_codebook).
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_codebook_chunked: cannot open " + path);
    char magic[8] = {'S', 'S', 'O', 'M', 'W', '1', '\0', '\0'};
    f.write(magic, 8);
    uint32_t hdr[4] = {cfg.map_rows, cfg.map_cols, V, 0};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));

    const uint32_t blk = 256;
    for (uint32_t v0 = 0; v0 < V; v0 += ch) {
        uint32_t nv = std::min(ch, V - v0);
        uint64_t n = (uint64_t)K * nv;
        uint32_t grd = static_cast<uint32_t>((n + blk - 1) / blk);
        h2f_kernel<<<grd, blk>>>(d_W_fm + (uint64_t)v0 * K, d_tmp, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_tmp, n * sizeof(float), cudaMemcpyDeviceToHost));
        f.write(reinterpret_cast<const char*>(h_buf.data()),
                (std::streamsize)(n * sizeof(float)));
    }
    cudaFree(d_tmp);
    if (!f) throw std::runtime_error("save_codebook_chunked: write failed for " + path);
    std::printf("saved codebook (chunked) -> %s  (%u neurons x %u features, %.1f GiB)\n",
                path.c_str(), K, V, (double)V * K * 4 / (1ull << 30));
}

void FeatureMajorCodebook::download_bmu1(std::vector<uint32_t>& out) const {
    out.resize(n_samples);                          // per-sample winning neuron (the clustering)
    CUDA_CHECK(cudaMemcpy(out.data(), d_bmu1, (size_t)n_samples * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

void FeatureMajorCodebook::resize_samples(uint32_t new_n) {
    if (new_n == n_samples) return;
    cudaFree(d_bmu1);     cudaFree(d_bmu2);
    cudaFree(d_bmu_prev); cudaFree(d_bmu_dist);
    n_samples = new_n;
    CUDA_CHECK(cudaMalloc(&d_bmu1,     (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu2,     (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu_prev, (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmu_dist, (size_t)n_samples * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bmu1,     0xFF, (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bmu2,     0xFF, (size_t)n_samples * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bmu_prev, 0xFF, (size_t)n_samples * sizeof(uint32_t)));
}

void FeatureMajorCodebook::upload_W_fm(const std::vector<float>& W) {
    const size_t expect = (size_t)n_features * n_neurons;
    if (W.size() != expect)
        throw std::runtime_error("upload_W_fm: weight count " + std::to_string(W.size()) +
                                 " != expected " + std::to_string(expect) +
                                 " (n_features * n_neurons)");
    // Copy float to device, convert to __half in-place.
    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, expect * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tmp, W.data(), expect * sizeof(float), cudaMemcpyHostToDevice));
    const uint32_t blk = 256;
    const uint32_t grd = static_cast<uint32_t>((expect + blk - 1) / blk);
    f2h_kernel<<<grd, blk>>>(d_tmp, d_W_fm, expect);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_tmp);
    recompute_norms();
}

void FeatureMajorCodebook::recompute_norms() {
    const uint32_t K   = n_neurons;              // neuron count
    const uint32_t blk = 256;                    // threads per block
    const uint32_t grd = (K + blk - 1) / blk;    // blocks needed to cover all neurons
    // d_norms: plain squared L2 norm (power 2) — used for the Euclidean BMU score and distance.
    norm_fm_kernel<<<grd, blk>>>(d_W_fm, d_norms, K, n_features, 2.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace sparsesom
