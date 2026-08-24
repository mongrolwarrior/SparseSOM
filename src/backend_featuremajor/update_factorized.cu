// update_factorized.cu — factorized batch-SOM update for the feature-major codebook.
//
// The brute-force SOM update costs O(neurons × samples × features-per-sample) and is
// independent of the neighbourhood width σ. This factorizes it into three cheap stages:
//
//   1. ACCUMULATE  — scatter each sample onto its winning neuron (BMU): count how many
//                    samples each neuron won (the "denominator" m) and sum their feature
//                    vectors (the "numerator" S). One atomicAdd per present feature.
//   2. BLUR        — spread each neuron's totals to its grid neighbours with a separable
//                    box filter (running sums). P box passes ≈ a Gaussian and the cost is
//                    O(map × features) regardless of σ — this is the σ-independence win.
//   3. WRITE       — the new weight of (neuron, feature) is blurred-numerator divided by
//                    blurred-denominator: W = Num / Den (the neighbourhood-weighted mean).
//
// "Coalesce memory transfers": the accumulate and blur keep the working tiles NODE-MAJOR
// ([neuron × feature]) so the blur's neighbouring threads touch contiguous memory. The
// codebook itself is FEATURE-MAJOR, so the final write transposes each tile through shared
// memory — making both the tile read and the codebook write coalesced. The numerator is
// also STREAMED in feature chunks so we never materialise the full [neurons × features]
// numerator in memory.
//
// Scope here: SQUARE lattice, PLANAR (clamped-at-edge) neighbourhood, full consolidated
// corpus (every sample every update — no mini-batches), every sample weighted equally.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Throw a clean error on a failed CUDA call (e.g. an out-of-memory tile alloc at a large map),
// rather than leaving a null pointer for a kernel to dereference into an illegal memory access.
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

namespace sparsesom {

// ── Stage 1: accumulate ─────────────────────────────────────────────────────────

// Denominator: count the samples that landed on each neuron.
//   bmu1  = per-sample winning neuron index [n_samples]
//   n_samples = number of samples
//   m     = output, one running count per neuron (denominator, pre-blur) [n_neurons]
static __global__ void accum_m_kernel(const uint32_t* __restrict__ bmu1,
                                      uint32_t n_samples, float* __restrict__ m) {
    uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;   // the sample this thread handles
    if (a >= n_samples) return;
    atomicAdd(&m[bmu1[a]], 1.0f);                          // +1 to this sample's winner
}

// Numerator for one feature chunk [f0, f0+ch_valid): for every sample, add 1 to its
// winning neuron's slot for each present feature that falls inside the chunk. (Samples
// are sparse-binary, so a present feature contributes exactly 1.)
//   row_ptr   = CSR row offsets [n_samples+1]; sample a's features are col_idx[row_ptr[a]..row_ptr[a+1])
//   col_idx   = CSR feature ids (the features present in each sample)
//   bmu1      = per-sample winning neuron index
//   n_samples     = number of samples
//   f0        = first global feature id of this chunk
//   ch_valid  = number of real features in this chunk (<= ch_stride; the bound we test)
//   ch_stride = allocated channel stride of S_tile (the tile's full width)
//   S_tile    = output node-major tile [n_neurons × ch_stride]; slot = S_tile[c*ch_stride + (f-f0)]
static __global__ void accum_chunk_kernel(const uint32_t* __restrict__ row_ptr,
                                          const uint16_t* __restrict__ col_idx,
                                          const uint32_t* __restrict__ bmu1,
                                          uint32_t n_samples, uint32_t f0,
                                          uint32_t ch_valid, uint32_t ch_stride,
                                          float* __restrict__ S_tile) {
    uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;   // the sample this thread handles
    if (a >= n_samples) return;
    uint32_t c = bmu1[a];                                 // this sample's winning neuron
    for (uint32_t j = row_ptr[a]; j < row_ptr[a + 1]; ++j) {
        uint32_t f = col_idx[j];                          // a present feature of sample a
        if (f >= f0 && f < f0 + ch_valid)                 // is it inside this chunk?
            atomicAdd(&S_tile[(size_t)c * ch_stride + (f - f0)], 1.0f);
    }
}

// Float numerator: identical to accum_chunk_kernel but adds each present feature's VALUE x_v
// (from `values`) rather than 1. The denominator (sample counts) is unchanged — the batch-SOM
// update is W = Σ_{winners} x_v / (count of winners), the neighbourhood-weighted feature mean.
static __global__ void accum_chunk_float_kernel(const uint32_t* __restrict__ row_ptr,
                                                const uint16_t* __restrict__ col_idx,
                                                const float*    __restrict__ values,
                                                const uint32_t* __restrict__ bmu1,
                                                uint32_t n_samples, uint32_t f0,
                                                uint32_t ch_valid, uint32_t ch_stride,
                                                float* __restrict__ S_tile) {
    uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_samples) return;
    uint32_t c = bmu1[a];
    for (uint32_t j = row_ptr[a]; j < row_ptr[a + 1]; ++j) {
        uint32_t f = col_idx[j];
        if (f >= f0 && f < f0 + ch_valid)
            atomicAdd(&S_tile[(size_t)c * ch_stride + (f - f0)], values[j]);
    }
}

// ── Stage 2: separable box blur (planar / clamped at the map edges) ──────────────
// Node-major data [rows × cols × C]: one thread per (line, channel) so a warp spans
// consecutive channels → coalesced. Running-sum trick makes each pass O(length) per line
// regardless of the radius. Horizontal pass blurs along columns; vertical along rows.
//   in/out = source / destination tiles, node-major [rows*cols*C]
//   rows   = map height, cols = map width (square lattice; both equal in practice)
//   C      = number of channels (features in this tile, or 1 for the denominator)
//   r      = box radius in cells (~σ); window is [c-r, c+r]
static __global__ void box_blur_h(const float* __restrict__ in, float* __restrict__ out,
                                  uint32_t rows, uint32_t cols, uint32_t C, int r) {
    uint32_t chan = blockIdx.x * blockDim.x + threadIdx.x;  // channel (feature) this thread owns
    uint32_t row  = blockIdx.y;                             // map row this thread owns
    if (chan >= C || row >= rows) return;
    const size_t base = (size_t)row * cols * C + chan;      // offset of (row, col=0, chan)
    const int    L    = (int)cols;                          // length of the axis being blurred
    float sum = 0.0f;                                       // running window sum
    for (int c = 0; c <= r && c < L; ++c) sum += in[base + (size_t)c * C];  // seed window at col 0
    for (int c = 0; c < L; ++c) {
        out[base + (size_t)c * C] = sum;                    // write current window sum
        int add = c + r + 1, rem = c - r;                   // cell entering / leaving the window
        if (add < L)  sum += in[base + (size_t)add * C];
        if (rem >= 0) sum -= in[base + (size_t)rem * C];
    }
}

static __global__ void box_blur_v(const float* __restrict__ in, float* __restrict__ out,
                                  uint32_t rows, uint32_t cols, uint32_t C, int r) {
    uint32_t chan = blockIdx.x * blockDim.x + threadIdx.x;  // channel (feature) this thread owns
    uint32_t col  = blockIdx.y;                             // map column this thread owns
    if (chan >= C || col >= cols) return;
    const size_t stride = (size_t)cols * C;                 // bytes (in floats) between rows
    const size_t base   = (size_t)col * C + chan;           // offset of (row=0, col, chan)
    const int    L      = (int)rows;                        // length of the axis being blurred
    float sum = 0.0f;                                       // running window sum
    for (int rr = 0; rr <= r && rr < L; ++rr) sum += in[base + (size_t)rr * stride];  // seed at row 0
    for (int rr = 0; rr < L; ++rr) {
        out[base + (size_t)rr * stride] = sum;              // write current window sum
        int add = rr + r + 1, rem = rr - r;                 // row entering / leaving the window
        if (add < L)  sum += in[base + (size_t)add * stride];
        if (rem >= 0) sum -= in[base + (size_t)rem * stride];
    }
}

// ── Stage 3: write (tiled transpose into the feature-major codebook) ─────────────
static constexpr uint32_t FM_TT = 32;   // transpose tile is FM_TT × FM_TT cells

// Transpose the blurred node-major chunk Num [n_neurons × ch_stride] into the feature-major
// codebook Wfm [features × K], folding in the 1/Den division and skipping neurons that won
// no samples (Den==0, left unchanged). Shared-memory tiling makes the Num read coalesced
// over features and the Wfm write coalesced over neurons.
//   Num       = blurred numerator tile, node-major [K × ch_stride]
//   Den       = blurred denominator [K]
//   Wfm       = feature-major codebook [V × K] being written
//   K         = neuron count (also Wfm's column stride)
//   ch_stride = Num's channel stride (tile width)
//   f0        = global feature id where this chunk starts
//   ch_valid  = number of real features in this chunk (<= ch_stride)
static __global__ void transpose_chunk_to_fm_kernel(const float* __restrict__ Num,
                                                    const float* __restrict__ Den,
                                                    __half* __restrict__ Wfm,
                                                    uint32_t K, uint32_t ch_stride,
                                                    uint32_t f0, uint32_t ch_valid) {
    __shared__ float tile[FM_TT][FM_TT + 1];        // +1 column pads away shared-mem bank conflicts
    const uint32_t n0 = blockIdx.x * FM_TT;         // first neuron of this tile
    const uint32_t l0 = blockIdx.y * FM_TT;         // first in-chunk feature of this tile

    // Read a [neuron × feature] block of Num (coalesced over features), scaled by 1/Den.
    uint32_t n_r = n0 + threadIdx.x;                // neuron this thread reads
    uint32_t l_r = l0 + threadIdx.y;                // in-chunk feature this thread reads
    if (n_r < K && l_r < ch_valid) {
        float d   = Den[n_r];
        float inv = (d > 1e-12f) ? 1.0f / d : 0.0f;
        tile[threadIdx.y][threadIdx.x] = Num[(size_t)n_r * ch_stride + l_r] * inv;
    }
    __syncthreads();

    // Write the transposed block to Wfm as __half (coalesced over neurons); skip dead neurons.
    uint32_t l_w = l0 + threadIdx.x;                // feature this thread writes
    uint32_t n_w = n0 + threadIdx.y;                // neuron this thread writes
    if (l_w < ch_valid && n_w < K && Den[n_w] > 1e-12f)
        Wfm[(size_t)(f0 + l_w) * K + n_w] = __float2half(tile[threadIdx.x][threadIdx.y]);
}

// ── Host launcher ────────────────────────────────────────────────────────────────
void launch_update_factorized(const DeviceDataset& dds, FeatureMajorCodebook& state,
                              float sigma, cudaStream_t stream) {
    const uint32_t K     = state.n_neurons;                 // neuron count (full square map)
    const uint32_t V     = state.n_features;                   // feature (features) count
    const uint32_t n_samples = state.n_samples;                // samples in the corpus
    const uint32_t rows  = state.cfg.map_rows;              // map height
    const uint32_t cols  = state.cfg.map_cols;              // map width
    const uint32_t CH    = std::min<uint32_t>(2048, V);     // feature-chunk width (tile channels)
    const int      passes = state.cfg.box_passes < 1 ? 1 : state.cfg.box_passes;  // box passes/axis
    const int      r      = (int)lroundf(sigma);            // box radius in cells (~σ)

    // Working tiles. S/tmp are node-major chunk tiles [K × CH]; m/tmp_m are per-neuron
    // denominators. tmp / tmp_m are scratch for the ping-pong blur passes.
    float *S = nullptr, *tmp = nullptr, *m = nullptr, *tmp_m = nullptr;
    CUDA_CHECK(cudaMalloc(&S,     (size_t)K * CH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp,   (size_t)K * CH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m,     (size_t)K      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_m, (size_t)K      * sizeof(float)));

    constexpr uint32_t BLK = 256;                           // threads per block for the 1-D kernels
    const uint32_t art_grid = (n_samples + BLK - 1) / BLK;      // blocks to cover all samples

    // Run `passes` horizontal then `passes` vertical box passes over C channels of `buf`,
    // using `scratch` as the ping-pong partner. 2·passes swaps is even, so the result lands
    // back in `buf`. `scratch` must hold K·C floats.
    auto blur_separable = [&](float* buf, float* scratch, uint32_t C) {
        float* a = buf;                                     // current source buffer
        float* b = scratch;                                 // current destination buffer
        const dim3 gh((C + BLK - 1) / BLK, rows);           // grid for horizontal pass
        const dim3 gv((C + BLK - 1) / BLK, cols);           // grid for vertical pass
        for (int p = 0; p < passes; ++p) {                  // horizontal passes
            box_blur_h<<<gh, BLK, 0, stream>>>(a, b, rows, cols, C, r);
            std::swap(a, b);
        }
        for (int p = 0; p < passes; ++p) {                  // vertical passes
            box_blur_v<<<gv, BLK, 0, stream>>>(a, b, rows, cols, C, r);
            std::swap(a, b);
        }
        if (a != buf)                                       // defensive: ensure result is in buf
            cudaMemcpyAsync(buf, a, (size_t)K * C * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
    };

    // Denominator: count samples per neuron, then blur the counts over the map.
    cudaMemsetAsync(m, 0, (size_t)K * sizeof(float), stream);
    accum_m_kernel<<<art_grid, BLK, 0, stream>>>(state.d_bmu1, n_samples, m);
    blur_separable(m, tmp_m, 1);

    // Numerator: stream the features in chunks of CH so the working tile stays small. The
    // tile is allocated with stride CH; the last (partial) chunk leaves its unused channels
    // zero from the memset, and accumulate/blur/write all use the same CH stride.
    for (uint32_t f0 = 0; f0 < V; f0 += CH) {
        const uint32_t ch = std::min(CH, V - f0);           // real features in this chunk
        cudaMemsetAsync(S, 0, (size_t)K * CH * sizeof(float), stream);
        if (dds.is_float())                                 // weighted numerator (Σ x_v)
            accum_chunk_float_kernel<<<art_grid, BLK, 0, stream>>>(
                dds.d_row_ptr, dds.d_col_idx, dds.d_values, state.d_bmu1, n_samples, f0, ch, CH, S);
        else                                                // binary numerator (Σ 1)
            accum_chunk_kernel<<<art_grid, BLK, 0, stream>>>(
                dds.d_row_ptr, dds.d_col_idx, state.d_bmu1, n_samples, f0, ch, CH, S);
        blur_separable(S, tmp, CH);
        dim3 blk(FM_TT, FM_TT);                             // transpose block: FM_TT × FM_TT threads
        dim3 grd((K + FM_TT - 1) / FM_TT, (ch + FM_TT - 1) / FM_TT);  // tiles over neurons × features
        transpose_chunk_to_fm_kernel<<<grd, blk, 0, stream>>>(
            S, m, state.d_W_fm, K, CH, f0, ch);
    }

    cudaStreamSynchronize(stream);
    cudaFree(S); cudaFree(tmp); cudaFree(m); cudaFree(tmp_m);
}

} // namespace sparsesom
