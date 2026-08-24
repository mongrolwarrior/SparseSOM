// kaski_lagus.cu — the Kaski–Lagus combined distortion measure (the default stop metric).
//
// For each monitored sample, KL = ‖x − w_bmu1‖ (the quantisation term, already in d_bmu_dist)
// + the length of the grid path from bmu1 to bmu2, summed as input-space edge distances
// ‖w_p − w_q‖ between consecutive neurons along the path. This folds quantisation accuracy and
// topology preservation into one continuous, input-space-distance scalar — so its epoch-to-epoch
// trace decays smoothly and supports a principled plateau stop rule (unlike a noisy topographic-
// error step function). The measure is the mean over the monitored samples.
//
// Path: a 4-connected L-path (step along rows, then columns), capped at kl_path_radius edges —
// bmu1 and bmu2 are almost always grid-adjacent, so the path is usually a single edge. Edge
// distance uses the PLAIN Euclidean norm (commensurable with the QE term): ‖w_p − w_q‖² =
// ‖w_p‖² + ‖w_q‖² − 2·(w_p·w_q), reusing the precomputed d_norms for the two ‖·‖² terms.

#include "sparsesom/feature_major_codebook.hpp"
#include <cuda_fp16.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

namespace sparsesom {

// Input-space distance between neurons p and q via the plain norm: sqrt(‖w_p‖²+‖w_q‖²−2 w_p·w_q).
// W_fm is feature-major [V × K] (column k is neuron k's weights, stride K); norms[k] = ‖w_k‖².
static __device__ float edge_dist(const __half* __restrict__ W_fm, const float* __restrict__ norms,
                                  uint32_t K, uint32_t V, uint32_t p, uint32_t q) {
    double dot = 0.0;
    for (uint32_t v = 0; v < V; ++v)
        dot += (double)__half2float(W_fm[(size_t)v * K + p]) *
               (double)__half2float(W_fm[(size_t)v * K + q]);
    double d2 = (double)norms[p] + (double)norms[q] - 2.0 * dot;
    return sqrtf((float)fmax(0.0, d2));
}

// One thread per monitored sample. Walks the capped L-path bmu1→bmu2 and writes the per-sample
// KL contribution (bmu_dist + path length) into out[a].
static __global__ void kaski_lagus_kernel(
    const __half* __restrict__ W_fm, const float* __restrict__ norms,
    const uint32_t* __restrict__ bmu1, const uint32_t* __restrict__ bmu2,
    const float* __restrict__ bmu_dist,
    uint32_t n_monitor, uint32_t K, uint32_t V, uint32_t rows, uint32_t cols,
    int path_radius, float* __restrict__ out)
{
    uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_monitor) return;

    const uint32_t c1 = bmu1[a], c2 = bmu2[a];
    int r1 = (int)(c1 / cols), q1 = (int)(c1 % cols);     // grid coords of the BMU
    int r2 = (int)(c2 / cols), q2 = (int)(c2 % cols);     // grid coords of the 2nd-BMU

    double path = 0.0;
    uint32_t prev = c1;
    int steps = 0, rr = r1, qq = q1;
    while (rr != r2 && steps < path_radius) {             // step along rows toward bmu2
        rr += (r2 > rr) ? 1 : -1;
        uint32_t cur = (uint32_t)rr * cols + (uint32_t)qq;
        path += edge_dist(W_fm, norms, K, V, prev, cur);
        prev = cur; ++steps;
    }
    while (qq != q2 && steps < path_radius) {             // then along columns
        qq += (q2 > qq) ? 1 : -1;
        uint32_t cur = (uint32_t)rr * cols + (uint32_t)qq;
        path += edge_dist(W_fm, norms, K, V, prev, cur);
        prev = cur; ++steps;
    }
    out[a] = bmu_dist[a] + (float)path;
}

double kaski_lagus(const FeatureMajorCodebook& s, cudaStream_t stream) {
    if (s.cfg.kl_path_score_weighted)
        throw std::runtime_error("kaski_lagus: score-weighted path norm not implemented; the QE "
                                 "term is plain Euclidean, so the plain norm is the commensurable "
                                 "default (set kl_path_score_weighted = false).");
    const uint32_t n_mon = s.cfg.kl_monitor_cap
                         ? std::min(s.cfg.kl_monitor_cap, s.n_samples) : s.n_samples;
    if (n_mon == 0) return 0.0;

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n_mon * sizeof(float)));
    const uint32_t blk = 128, grd = (n_mon + blk - 1) / blk;
    kaski_lagus_kernel<<<grd, blk, 0, stream>>>(
        s.d_W_fm, s.d_norms, s.d_bmu1, s.d_bmu2, s.d_bmu_dist,
        n_mon, s.n_neurons, s.n_features, s.cfg.map_rows, s.cfg.map_cols,
        (int)s.cfg.kl_path_radius, d_out);
    cudaStreamSynchronize(stream);

    double sum = thrust::reduce(thrust::device, d_out, d_out + n_mon, 0.0, thrust::plus<double>());
    cudaFree(d_out);
    return sum / (double)n_mon;       // mean KL distortion over the monitored samples
}

} // namespace sparsesom
