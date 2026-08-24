// bmu.cu — best-match-unit (BMU) computation against the feature-major codebook.
//
// For every input sample, find the neuron whose weight vector best matches the sample's
// sparse feature set, and record it in d_bmu1 along with the Euclidean distance in
// d_bmu_dist. This is the "assignment" half of a SOM training step.
//
// Matching is plain (squared) Euclidean: the BMU is argmin‖x − w‖². Expanding,
// ‖x − w‖² = ‖x‖² − 2·(x·w) + ‖w‖², and ‖x‖² is a per-sample constant, so the BMU is
//     argmax_k [ 2·(x·w_k) − ‖w_k‖² ]   with ‖w_k‖² = d_norms[k].
// (An earlier "precision-weighted" score with an exponent β was removed: β>0 weights each
//  feature's distortion by the neuron's own weight there, which lets low-weight neurons dodge
//  the penalty for missing a sample's features and lets negative weights cheat the score —
//  collapsing assignments onto a few neurons. β=0/Euclidean is the only sound choice here.)
//
// TWO data modes (same maths, same tiled algorithm; selected by DeviceDataset::is_float()):
//   BINARY — x_v ≡ 1: x·w_k = Σ_v w[v][k] over the present features (gather-SUM); ‖x‖² = nnz.
//   FLOAT  — x_v are weights: x·w_k = Σ_v x_v·w[v][k] (gather-MULTIPLY); ‖x‖² = Σ x_v² (d_xnorm2).
//   Binary is the exact special case of float with all values 1, so the two kernels share their
//   structure; the binary kernel is kept separate (and untouched) so its hot path carries no
//   per-feature value load or multiply.
//
// Algorithm (extracted from MedSOM-Naive's SpmmTile bmu_spmm.cu), one GPU block per tile
// of SPMM_TA samples:
//   1. Load each sample's feature list (and, in float mode, its values) into shared memory.
//   2. Build a de-duplicated feature set for the tile plus, per distinct feature, a bitmask
//      of which samples in the tile have it (and, in float mode, each such sample's value).
//   3. All threads sweep neurons k = tid, tid+BLK, …; for each distinct feature they load
//      W_fm[v*K + k] ONCE (coalesced across the warp) and add it (×value, in float mode) into
//      every tile sample that has that feature — so the codebook read is shared across the
//      whole tile. Each thread keeps a running best (score, neuron, dot) per sample.
//   4. Warp-level then block-level argmax reduction.
//   5. Thread 0 writes d_bmu1 and d_bmu_dist for the tile's live samples.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <cstdio>

namespace sparsesom {

static constexpr uint32_t SPMM_TA       = 16;   // samples handled per tile (one block)
static constexpr uint32_t SPMM_NNZ_MAX  = 20;   // max features per sample considered (extras dropped)
static constexpr uint32_t SPMM_BLK      = 256;  // threads per block
static constexpr uint32_t SPMM_WARPS    = SPMM_BLK / 32;             // warps per block (8)
static constexpr uint32_t SPMM_MAX_DIST = SPMM_TA * SPMM_NNZ_MAX;    // max distinct features/tile (320)

// ── Binary BMU (gather-sum; ‖x‖² = nnz) ─────────────────────────────────────────────
static __global__ void bmu_spmm_kernel(
    const uint32_t* __restrict__ d_row_ptr,      // CSR row offsets [n_samples+1]
    const uint16_t* __restrict__ d_col_idx,      // CSR feature ids (present features per sample)
    const __half*   __restrict__ d_W_fm,         // feature-major codebook [V × K], FP16
    const float*    __restrict__ d_norms,        // ‖w‖² per neuron [K]
    uint32_t* __restrict__ d_bmu1,               // out: best-matching neuron per sample [n_samples]
    uint32_t* __restrict__ d_bmu2,               // out: second-best neuron per sample [n_samples]
    float*    __restrict__ d_bmu_dist,           // out: Euclidean distance to bmu1 [n_samples]
    uint32_t  n_samples,                         // number of samples
    uint32_t  N)                                 // neuron count (K); also the W_fm column stride
{
    // ── Shared state for this tile ──────────────────────────────────────────────
    __shared__ uint32_t s_arts[SPMM_TA];                 // global sample index per tile slot
    __shared__ uint32_t s_nnz [SPMM_TA];                 // feature count per tile slot (capped)
    __shared__ uint16_t s_col [SPMM_TA][SPMM_NNZ_MAX];   // each slot's feature ids

    __shared__ uint16_t s_distinct[SPMM_MAX_DIST];       // the tile's distinct feature ids
    __shared__ uint16_t s_mask    [SPMM_MAX_DIST];       // bit a set ⇔ tile sample a has this feature
    __shared__ uint32_t s_n_dist;                        // number of distinct features in the tile

    // Per-warp top-2 (score, neuron) for each sample, for the block-level reduction. The dot
    // (Σ w) is tracked only for the best, since the reported distance is to bmu1.
    __shared__ float    s_score [SPMM_TA][SPMM_WARPS];   // best score
    __shared__ uint32_t s_idx   [SPMM_TA][SPMM_WARPS];   // neuron achieving it (bmu1)
    __shared__ float    s_dot   [SPMM_TA][SPMM_WARPS];   // Σ w at the best neuron (for the distance)
    __shared__ float    s_score2[SPMM_TA][SPMM_WARPS];   // second-best score
    __shared__ uint32_t s_idx2  [SPMM_TA][SPMM_WARPS];   // neuron achieving it (bmu2)

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;          // lane within the warp
    const uint32_t warp = tid >> 5u;          // warp index within the block

    const uint32_t tile_start = blockIdx.x * SPMM_TA;     // first sample of this tile

    // ── Phase 1: load each sample's feature list into shared memory ─────────────
    if (tid < SPMM_TA) {
        uint32_t pos = tile_start + tid;                  // global sample index for this slot
        if (pos < n_samples) {
            s_arts[tid]  = pos;
            uint32_t s   = d_row_ptr[pos];                // start of this sample's CSR row
            uint32_t e   = d_row_ptr[pos + 1];            // end of it
            uint32_t nnz = e - s;                         // how many features it has
            if (nnz > SPMM_NNZ_MAX) nnz = SPMM_NNZ_MAX;   // cap (extra features ignored)
            s_nnz[tid] = nnz;
            for (uint32_t j = 0; j < nnz; ++j)
                s_col[tid][j] = d_col_idx[s + j];
        } else {                                          // padding slot in the last partial tile
            s_arts[tid] = 0;
            s_nnz[tid]  = 0;
        }
    }
    __syncthreads();

    // How many real samples this tile holds (the last tile may be partial).
    const uint32_t n_live = (n_samples - tile_start < SPMM_TA) ? (n_samples - tile_start) : SPMM_TA;

    // ── Phase 2: de-duplicate the tile's features + record per-feature membership ─
    // Thread 0 only; the lists total ≤160 entries so this is negligible.
    if (tid == 0) {
        uint32_t n_dist = 0;                              // distinct features found so far
        for (uint32_t a = 0; a < n_live; ++a) {
            for (uint32_t j = 0; j < s_nnz[a]; ++j) {
                uint16_t f = s_col[a][j];                 // a feature of sample a
                bool found = false;
                for (uint32_t d = 0; d < n_dist; ++d) {   // already seen this feature?
                    if (s_distinct[d] == f) {
                        s_mask[d] |= (uint16_t)(1u << a);  // mark sample a as having it
                        found = true;
                        break;
                    }
                }
                if (!found) {                             // new distinct feature
                    s_distinct[n_dist] = f;
                    s_mask[n_dist]     = (uint16_t)(1u << a);
                    ++n_dist;
                }
            }
        }
        s_n_dist = n_dist;
    }
    __syncthreads();

    const uint32_t n_dist = s_n_dist;

    // ── Phase 3: sweep neurons; keep a per-sample running top-2 ─────────────────
    // Each thread scans a disjoint set of neurons (k = tid, tid+BLK, …), so within and across
    // threads every candidate neuron is distinct — no duplicate-index handling is needed when
    // merging top-2 sets later.
    float    l_score [SPMM_TA];   // best score this thread has seen, per sample
    uint32_t l_idx   [SPMM_TA];   // neuron achieving it (bmu1)
    float    l_dot   [SPMM_TA];   // Σ w at the best neuron (for the Euclidean distance)
    float    l_score2[SPMM_TA];   // second-best score
    uint32_t l_idx2  [SPMM_TA];   // neuron achieving it (bmu2)
    #pragma unroll
    for (uint32_t a = 0; a < SPMM_TA; ++a) {
        l_score[a] = -FLT_MAX; l_idx[a] = 0; l_dot[a] = 0.0f;
        l_score2[a] = -FLT_MAX; l_idx2[a] = 0;
    }

    for (uint32_t k = tid; k < N; k += SPMM_BLK) {        // this thread's neurons
        float std_dot[SPMM_TA] = {};                      // Σ_v w over present features (= x·w)

        for (uint32_t d = 0; d < n_dist; ++d) {
            float    w   = __half2float(d_W_fm[(size_t)s_distinct[d] * N + k]);  // FP16→FP32, coalesced
            uint16_t msk = s_mask[d];
            #pragma unroll
            for (uint32_t a = 0; a < SPMM_TA; ++a)
                if (msk & (uint16_t)(1u << a)) std_dot[a] += w;
        }

        float nrm = d_norms[k];
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            if (a >= n_live) continue;
            // argmax(2·x·w − ‖w‖²) == argmin‖x − w‖² (the ‖x‖² term is a per-sample constant).
            float score = 2.0f * std_dot[a] - nrm;
            if (score > l_score[a]) {           // new best → demote old best to second
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = score;      l_idx[a]  = k;  l_dot[a] = std_dot[a];
            } else if (score > l_score2[a]) {   // new second-best
                l_score2[a] = score;      l_idx2[a] = k;
            }
        }
    }

    // ── Phase 4: warp-level top-2 reduction (fold the 32 lanes into lane 0) ─────
    // Merge each incoming lane's top-2 (ob1≥ob2) into this lane's (l_score≥l_score2). All four
    // candidate neurons are distinct (disjoint k per lane), so this is a plain top-2 merge.
    #pragma unroll
    for (uint32_t a = 0; a < SPMM_TA; ++a) {
        for (int off = 16; off >= 1; off >>= 1) {
            float    ob1 = __shfl_down_sync(0xFFFFFFFFu, l_score [a], off);
            uint32_t oi1 = __shfl_down_sync(0xFFFFFFFFu, l_idx   [a], off);
            float    od1 = __shfl_down_sync(0xFFFFFFFFu, l_dot   [a], off);
            float    ob2 = __shfl_down_sync(0xFFFFFFFFu, l_score2[a], off);
            uint32_t oi2 = __shfl_down_sync(0xFFFFFFFFu, l_idx2  [a], off);
            if (ob1 > l_score[a]) {                       // incoming best becomes the new best
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = ob1;        l_idx[a]  = oi1; l_dot[a] = od1;
            } else if (ob1 > l_score2[a]) {
                l_score2[a] = ob1;        l_idx2[a] = oi1;
            }
            if (ob2 > l_score2[a]) {                      // incoming second-best may still place
                l_score2[a] = ob2;        l_idx2[a] = oi2;
            }
        }
    }
    if (lane == 0) {                                      // each warp's lane 0 publishes its top-2
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            s_score [a][warp] = l_score [a];  s_idx [a][warp] = l_idx [a];  s_dot[a][warp] = l_dot[a];
            s_score2[a][warp] = l_score2[a];  s_idx2[a][warp] = l_idx2[a];
        }
    }
    __syncthreads();

    // ── Phase 5: thread 0 reduces top-2 across warps and writes the outputs ─────
    if (tid == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            if (a >= n_live) continue;
            float    b1 = s_score [a][0];  uint32_t i1 = s_idx [a][0];  float d1 = s_dot[a][0];
            float    b2 = s_score2[a][0];  uint32_t i2 = s_idx2[a][0];
            for (uint32_t w = 1; w < SPMM_WARPS; ++w) {   // same top-2 merge, across warps
                float ob1 = s_score [a][w]; uint32_t oi1 = s_idx [a][w]; float od1 = s_dot[a][w];
                float ob2 = s_score2[a][w]; uint32_t oi2 = s_idx2[a][w];
                if (ob1 > b1) { b2 = b1; i2 = i1; b1 = ob1; i1 = oi1; d1 = od1; }
                else if (ob1 > b2) { b2 = ob1; i2 = oi1; }
                if (ob2 > b2) { b2 = ob2; i2 = oi2; }
            }
            uint32_t art = s_arts[a];
            uint32_t nnz = s_nnz [a];                     // ‖x‖² for a binary sample = feature count
            d_bmu1[art]     = i1;
            d_bmu2[art]     = i2;
            // ‖x − w‖ = sqrt(‖x‖² − 2·x·w + ‖w‖²); clamp tiny negatives from rounding.
            d_bmu_dist[art] = sqrtf(fmaxf(0.0f, (float)nnz - 2.0f * d1 + d_norms[i1]));
        }
    }
}

// ── Float BMU (gather-multiply; ‖x‖² = Σ x_v² from d_xnorm2) ─────────────────────────
// Same tiled algorithm as the binary kernel; the only differences are: each sample also loads
// its per-feature values; the de-dup pass records, per (distinct feature, sample), that sample's
// value; the neuron sweep accumulates value·w instead of w; and the distance uses the sample's
// precomputed Σ x_v² instead of nnz.
static __global__ void bmu_spmm_float_kernel(
    const uint32_t* __restrict__ d_row_ptr,
    const uint16_t* __restrict__ d_col_idx,
    const float*    __restrict__ d_values,       // per-nonzero feature weights [n_nonzeros]
    const float*    __restrict__ d_xnorm2,       // Σ x_v² per sample [n_samples]
    const __half*   __restrict__ d_W_fm,         // feature-major codebook [V × K], FP16
    const float*    __restrict__ d_norms,
    uint32_t* __restrict__ d_bmu1,
    uint32_t* __restrict__ d_bmu2,
    float*    __restrict__ d_bmu_dist,
    uint32_t  n_samples,
    uint32_t  N)
{
    __shared__ uint32_t s_arts[SPMM_TA];
    __shared__ uint32_t s_nnz [SPMM_TA];
    __shared__ uint16_t s_col [SPMM_TA][SPMM_NNZ_MAX];
    __shared__ float    s_vcol[SPMM_TA][SPMM_NNZ_MAX];   // each slot's per-feature values

    __shared__ uint16_t s_distinct[SPMM_MAX_DIST];
    __shared__ uint16_t s_mask    [SPMM_MAX_DIST];
    __shared__ float    s_val     [SPMM_MAX_DIST][SPMM_TA];  // value of distinct feature d for sample a
    __shared__ uint32_t s_n_dist;

    __shared__ float    s_score [SPMM_TA][SPMM_WARPS];
    __shared__ uint32_t s_idx   [SPMM_TA][SPMM_WARPS];
    __shared__ float    s_dot   [SPMM_TA][SPMM_WARPS];
    __shared__ float    s_score2[SPMM_TA][SPMM_WARPS];
    __shared__ uint32_t s_idx2  [SPMM_TA][SPMM_WARPS];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t tile_start = blockIdx.x * SPMM_TA;

    // ── Phase 1: load each sample's feature ids AND values ──────────────────────
    if (tid < SPMM_TA) {
        uint32_t pos = tile_start + tid;
        if (pos < n_samples) {
            s_arts[tid]  = pos;
            uint32_t s   = d_row_ptr[pos];
            uint32_t e   = d_row_ptr[pos + 1];
            uint32_t nnz = e - s;
            if (nnz > SPMM_NNZ_MAX) nnz = SPMM_NNZ_MAX;
            s_nnz[tid] = nnz;
            for (uint32_t j = 0; j < nnz; ++j) {
                s_col [tid][j] = d_col_idx[s + j];
                s_vcol[tid][j] = d_values [s + j];
            }
        } else {
            s_arts[tid] = 0;
            s_nnz[tid]  = 0;
        }
    }
    __syncthreads();

    const uint32_t n_live = (n_samples - tile_start < SPMM_TA) ? (n_samples - tile_start) : SPMM_TA;

    // ── Phase 2: de-duplicate features + record per-feature membership and value ─
    if (tid == 0) {
        uint32_t n_dist = 0;
        for (uint32_t a = 0; a < n_live; ++a) {
            for (uint32_t j = 0; j < s_nnz[a]; ++j) {
                uint16_t f = s_col [a][j];
                float    v = s_vcol[a][j];                 // this feature's value for sample a
                bool found = false;
                for (uint32_t d = 0; d < n_dist; ++d) {
                    if (s_distinct[d] == f) {
                        s_mask[d] |= (uint16_t)(1u << a);
                        s_val [d][a] = v;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    s_distinct[n_dist] = f;
                    s_mask[n_dist]     = (uint16_t)(1u << a);
                    s_val [n_dist][a]  = v;
                    ++n_dist;
                }
            }
        }
        s_n_dist = n_dist;
    }
    __syncthreads();

    const uint32_t n_dist = s_n_dist;

    // ── Phase 3: sweep neurons; accumulate value·w per sample ───────────────────
    float    l_score [SPMM_TA];
    uint32_t l_idx   [SPMM_TA];
    float    l_dot   [SPMM_TA];
    float    l_score2[SPMM_TA];
    uint32_t l_idx2  [SPMM_TA];
    #pragma unroll
    for (uint32_t a = 0; a < SPMM_TA; ++a) {
        l_score[a] = -FLT_MAX; l_idx[a] = 0; l_dot[a] = 0.0f;
        l_score2[a] = -FLT_MAX; l_idx2[a] = 0;
    }

    for (uint32_t k = tid; k < N; k += SPMM_BLK) {
        float std_dot[SPMM_TA] = {};                      // Σ_v x_v·w over present features (= x·w)

        for (uint32_t d = 0; d < n_dist; ++d) {
            float    w   = __half2float(d_W_fm[(size_t)s_distinct[d] * N + k]);  // FP16→FP32, coalesced
            uint16_t msk = s_mask[d];
            #pragma unroll
            for (uint32_t a = 0; a < SPMM_TA; ++a)
                if (msk & (uint16_t)(1u << a)) std_dot[a] += s_val[d][a] * w;
        }

        float nrm = d_norms[k];
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            if (a >= n_live) continue;
            float score = 2.0f * std_dot[a] - nrm;
            if (score > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = score;      l_idx[a]  = k;  l_dot[a] = std_dot[a];
            } else if (score > l_score2[a]) {
                l_score2[a] = score;      l_idx2[a] = k;
            }
        }
    }

    // ── Phase 4: warp-level top-2 reduction ─────────────────────────────────────
    #pragma unroll
    for (uint32_t a = 0; a < SPMM_TA; ++a) {
        for (int off = 16; off >= 1; off >>= 1) {
            float    ob1 = __shfl_down_sync(0xFFFFFFFFu, l_score [a], off);
            uint32_t oi1 = __shfl_down_sync(0xFFFFFFFFu, l_idx   [a], off);
            float    od1 = __shfl_down_sync(0xFFFFFFFFu, l_dot   [a], off);
            float    ob2 = __shfl_down_sync(0xFFFFFFFFu, l_score2[a], off);
            uint32_t oi2 = __shfl_down_sync(0xFFFFFFFFu, l_idx2  [a], off);
            if (ob1 > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = ob1;        l_idx[a]  = oi1; l_dot[a] = od1;
            } else if (ob1 > l_score2[a]) {
                l_score2[a] = ob1;        l_idx2[a] = oi1;
            }
            if (ob2 > l_score2[a]) {
                l_score2[a] = ob2;        l_idx2[a] = oi2;
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            s_score [a][warp] = l_score [a];  s_idx [a][warp] = l_idx [a];  s_dot[a][warp] = l_dot[a];
            s_score2[a][warp] = l_score2[a];  s_idx2[a][warp] = l_idx2[a];
        }
    }
    __syncthreads();

    // ── Phase 5: reduce across warps; write outputs (distance uses Σ x_v²) ───────
    if (tid == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < SPMM_TA; ++a) {
            if (a >= n_live) continue;
            float    b1 = s_score [a][0];  uint32_t i1 = s_idx [a][0];  float d1 = s_dot[a][0];
            float    b2 = s_score2[a][0];  uint32_t i2 = s_idx2[a][0];
            for (uint32_t w = 1; w < SPMM_WARPS; ++w) {
                float ob1 = s_score [a][w]; uint32_t oi1 = s_idx [a][w]; float od1 = s_dot[a][w];
                float ob2 = s_score2[a][w]; uint32_t oi2 = s_idx2[a][w];
                if (ob1 > b1) { b2 = b1; i2 = i1; b1 = ob1; i1 = oi1; d1 = od1; }
                else if (ob1 > b2) { b2 = ob1; i2 = oi1; }
                if (ob2 > b2) { b2 = ob2; i2 = oi2; }
            }
            uint32_t art = s_arts[a];
            d_bmu1[art]     = i1;
            d_bmu2[art]     = i2;
            d_bmu_dist[art] = sqrtf(fmaxf(0.0f, d_xnorm2[art] - 2.0f * d1 + d_norms[i1]));
        }
    }
}

// ── Tile-32 constants ────────────────────────────────────────────────────────
static constexpr uint32_t TA32           = 32;
static constexpr uint32_t TA32_NNZ_MAX   = SPMM_NNZ_MAX;            // same cap
static constexpr uint32_t TA32_MAX_DIST  = TA32 * TA32_NNZ_MAX;     // 640
static constexpr uint32_t TA32_BLK       = SPMM_BLK;                // 256
static constexpr uint32_t TA32_WARPS     = TA32_BLK / 32;           // 8

// ── Binary BMU tile-32 (gather-sum; ‖x‖² = nnz) ─────────────────────────────
// Same algorithm as bmu_spmm_kernel but with 32 samples per tile and uint32_t
// masks, doubling codebook reuse per HBM load.
static __global__ void bmu_spmm_kernel_ta32(
    const uint32_t* __restrict__ d_row_ptr,
    const uint16_t* __restrict__ d_col_idx,
    const __half*   __restrict__ d_W_fm,
    const float*    __restrict__ d_norms,
    uint32_t* __restrict__ d_bmu1,
    uint32_t* __restrict__ d_bmu2,
    float*    __restrict__ d_bmu_dist,
    uint32_t  n_samples,
    uint32_t  N)
{
    __shared__ uint32_t s_arts[TA32];
    __shared__ uint32_t s_nnz [TA32];
    __shared__ uint16_t s_col [TA32][TA32_NNZ_MAX];

    __shared__ uint16_t s_distinct[TA32_MAX_DIST];
    __shared__ uint32_t s_mask    [TA32_MAX_DIST];   // uint32_t for 32 samples
    __shared__ uint32_t s_n_dist;

    __shared__ float    s_score [TA32][TA32_WARPS];
    __shared__ uint32_t s_idx   [TA32][TA32_WARPS];
    __shared__ float    s_dot   [TA32][TA32_WARPS];
    __shared__ float    s_score2[TA32][TA32_WARPS];
    __shared__ uint32_t s_idx2  [TA32][TA32_WARPS];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t tile_start = blockIdx.x * TA32;

    // ── Phase 1: load samples ──────────────────────────────────────────────
    if (tid < TA32) {
        uint32_t pos = tile_start + tid;
        if (pos < n_samples) {
            s_arts[tid] = pos;
            uint32_t s  = d_row_ptr[pos];
            uint32_t e  = d_row_ptr[pos + 1];
            uint32_t nnz = e - s;
            if (nnz > TA32_NNZ_MAX) nnz = TA32_NNZ_MAX;
            s_nnz[tid] = nnz;
            for (uint32_t j = 0; j < nnz; ++j)
                s_col[tid][j] = d_col_idx[s + j];
        } else {
            s_arts[tid] = 0;
            s_nnz[tid]  = 0;
        }
    }
    __syncthreads();

    const uint32_t n_live = (n_samples - tile_start < TA32) ? (n_samples - tile_start) : TA32;

    // ── Phase 2: de-duplicate features + membership masks ──────────────────
    if (tid == 0) {
        uint32_t n_dist = 0;
        for (uint32_t a = 0; a < n_live; ++a) {
            for (uint32_t j = 0; j < s_nnz[a]; ++j) {
                uint16_t f = s_col[a][j];
                bool found = false;
                for (uint32_t d = 0; d < n_dist; ++d) {
                    if (s_distinct[d] == f) {
                        s_mask[d] |= (1u << a);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    s_distinct[n_dist] = f;
                    s_mask[n_dist]     = (1u << a);
                    ++n_dist;
                }
            }
        }
        s_n_dist = n_dist;
    }
    __syncthreads();

    const uint32_t n_dist = s_n_dist;

    // ── Phase 3: sweep neurons; per-sample running top-2 ───────────────────
    float    l_score [TA32];
    uint32_t l_idx   [TA32];
    float    l_dot   [TA32];
    float    l_score2[TA32];
    uint32_t l_idx2  [TA32];
    #pragma unroll
    for (uint32_t a = 0; a < TA32; ++a) {
        l_score[a] = -FLT_MAX; l_idx[a] = 0; l_dot[a] = 0.0f;
        l_score2[a] = -FLT_MAX; l_idx2[a] = 0;
    }

    for (uint32_t k = tid; k < N; k += TA32_BLK) {
        float std_dot[TA32] = {};

        for (uint32_t d = 0; d < n_dist; ++d) {
            float    w   = __half2float(d_W_fm[(size_t)s_distinct[d] * N + k]);
            uint32_t msk = s_mask[d];
            #pragma unroll
            for (uint32_t a = 0; a < TA32; ++a)
                if (msk & (1u << a)) std_dot[a] += w;
        }

        float nrm = d_norms[k];
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            if (a >= n_live) continue;
            float score = 2.0f * std_dot[a] - nrm;
            if (score > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = score;      l_idx[a]  = k;  l_dot[a] = std_dot[a];
            } else if (score > l_score2[a]) {
                l_score2[a] = score;      l_idx2[a] = k;
            }
        }
    }

    // ── Phase 4: warp-level top-2 reduction ─────────────────────────────────
    #pragma unroll
    for (uint32_t a = 0; a < TA32; ++a) {
        for (int off = 16; off >= 1; off >>= 1) {
            float    ob1 = __shfl_down_sync(0xFFFFFFFFu, l_score [a], off);
            uint32_t oi1 = __shfl_down_sync(0xFFFFFFFFu, l_idx   [a], off);
            float    od1 = __shfl_down_sync(0xFFFFFFFFu, l_dot   [a], off);
            float    ob2 = __shfl_down_sync(0xFFFFFFFFu, l_score2[a], off);
            uint32_t oi2 = __shfl_down_sync(0xFFFFFFFFu, l_idx2  [a], off);
            if (ob1 > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = ob1;        l_idx[a]  = oi1; l_dot[a] = od1;
            } else if (ob1 > l_score2[a]) {
                l_score2[a] = ob1;        l_idx2[a] = oi1;
            }
            if (ob2 > l_score2[a]) {
                l_score2[a] = ob2;        l_idx2[a] = oi2;
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            s_score [a][warp] = l_score [a];  s_idx [a][warp] = l_idx [a];  s_dot[a][warp] = l_dot[a];
            s_score2[a][warp] = l_score2[a];  s_idx2[a][warp] = l_idx2[a];
        }
    }
    __syncthreads();

    // ── Phase 5: block-level reduction; write outputs ───────────────────────
    if (tid == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            if (a >= n_live) continue;
            float    b1 = s_score [a][0];  uint32_t i1 = s_idx [a][0];  float d1 = s_dot[a][0];
            float    b2 = s_score2[a][0];  uint32_t i2 = s_idx2[a][0];
            for (uint32_t w = 1; w < TA32_WARPS; ++w) {
                float ob1 = s_score [a][w]; uint32_t oi1 = s_idx [a][w]; float od1 = s_dot[a][w];
                float ob2 = s_score2[a][w]; uint32_t oi2 = s_idx2[a][w];
                if (ob1 > b1) { b2 = b1; i2 = i1; b1 = ob1; i1 = oi1; d1 = od1; }
                else if (ob1 > b2) { b2 = ob1; i2 = oi1; }
                if (ob2 > b2) { b2 = ob2; i2 = oi2; }
            }
            uint32_t art = s_arts[a];
            uint32_t nnz = s_nnz [a];
            d_bmu1[art]     = i1;
            d_bmu2[art]     = i2;
            d_bmu_dist[art] = sqrtf(fmaxf(0.0f, (float)nnz - 2.0f * d1 + d_norms[i1]));
        }
    }
}

// ── Float BMU tile-32 (gather-multiply; ‖x‖² = Σ x_v² from d_xnorm2) ────────
static __global__ void bmu_spmm_float_kernel_ta32(
    const uint32_t* __restrict__ d_row_ptr,
    const uint16_t* __restrict__ d_col_idx,
    const float*    __restrict__ d_values,
    const float*    __restrict__ d_xnorm2,
    const __half*   __restrict__ d_W_fm,
    const float*    __restrict__ d_norms,
    uint32_t* __restrict__ d_bmu1,
    uint32_t* __restrict__ d_bmu2,
    float*    __restrict__ d_bmu_dist,
    uint32_t  n_samples,
    uint32_t  N)
{
    __shared__ uint32_t s_arts[TA32];
    __shared__ uint32_t s_nnz [TA32];
    __shared__ uint16_t s_col [TA32][TA32_NNZ_MAX];
    __shared__ float    s_vcol[TA32][TA32_NNZ_MAX];

    __shared__ uint16_t s_distinct[TA32_MAX_DIST];
    __shared__ uint32_t s_mask    [TA32_MAX_DIST];
    // s_val lives in dynamic shared memory (80 KB) to stay under the 48 KB static limit
    extern __shared__ float s_val[];   // [TA32_MAX_DIST * TA32], row-major: s_val[d*TA32 + a]
    __shared__ uint32_t s_n_dist;

    __shared__ float    s_score [TA32][TA32_WARPS];
    __shared__ uint32_t s_idx   [TA32][TA32_WARPS];
    __shared__ float    s_dot   [TA32][TA32_WARPS];
    __shared__ float    s_score2[TA32][TA32_WARPS];
    __shared__ uint32_t s_idx2  [TA32][TA32_WARPS];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t tile_start = blockIdx.x * TA32;

    if (tid < TA32) {
        uint32_t pos = tile_start + tid;
        if (pos < n_samples) {
            s_arts[tid] = pos;
            uint32_t s  = d_row_ptr[pos];
            uint32_t e  = d_row_ptr[pos + 1];
            uint32_t nnz = e - s;
            if (nnz > TA32_NNZ_MAX) nnz = TA32_NNZ_MAX;
            s_nnz[tid] = nnz;
            for (uint32_t j = 0; j < nnz; ++j) {
                s_col [tid][j] = d_col_idx[s + j];
                s_vcol[tid][j] = d_values [s + j];
            }
        } else {
            s_arts[tid] = 0;
            s_nnz[tid]  = 0;
        }
    }
    __syncthreads();

    const uint32_t n_live = (n_samples - tile_start < TA32) ? (n_samples - tile_start) : TA32;

    if (tid == 0) {
        uint32_t n_dist = 0;
        for (uint32_t a = 0; a < n_live; ++a) {
            for (uint32_t j = 0; j < s_nnz[a]; ++j) {
                uint16_t f = s_col [a][j];
                float    v = s_vcol[a][j];
                bool found = false;
                for (uint32_t d = 0; d < n_dist; ++d) {
                    if (s_distinct[d] == f) {
                        s_mask[d] |= (1u << a);
                        s_val[d * TA32 + a] = v;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    s_distinct[n_dist] = f;
                    s_mask[n_dist]     = (1u << a);
                    s_val[n_dist * TA32 + a] = v;
                    ++n_dist;
                }
            }
        }
        s_n_dist = n_dist;
    }
    __syncthreads();

    const uint32_t n_dist = s_n_dist;

    float    l_score [TA32];
    uint32_t l_idx   [TA32];
    float    l_dot   [TA32];
    float    l_score2[TA32];
    uint32_t l_idx2  [TA32];
    #pragma unroll
    for (uint32_t a = 0; a < TA32; ++a) {
        l_score[a] = -FLT_MAX; l_idx[a] = 0; l_dot[a] = 0.0f;
        l_score2[a] = -FLT_MAX; l_idx2[a] = 0;
    }

    for (uint32_t k = tid; k < N; k += TA32_BLK) {
        float std_dot[TA32] = {};

        for (uint32_t d = 0; d < n_dist; ++d) {
            float    w   = __half2float(d_W_fm[(size_t)s_distinct[d] * N + k]);
            uint32_t msk = s_mask[d];
            #pragma unroll
            for (uint32_t a = 0; a < TA32; ++a)
                if (msk & (1u << a)) std_dot[a] += s_val[d * TA32 + a] * w;
        }

        float nrm = d_norms[k];
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            if (a >= n_live) continue;
            float score = 2.0f * std_dot[a] - nrm;
            if (score > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = score;      l_idx[a]  = k;  l_dot[a] = std_dot[a];
            } else if (score > l_score2[a]) {
                l_score2[a] = score;      l_idx2[a] = k;
            }
        }
    }

    #pragma unroll
    for (uint32_t a = 0; a < TA32; ++a) {
        for (int off = 16; off >= 1; off >>= 1) {
            float    ob1 = __shfl_down_sync(0xFFFFFFFFu, l_score [a], off);
            uint32_t oi1 = __shfl_down_sync(0xFFFFFFFFu, l_idx   [a], off);
            float    od1 = __shfl_down_sync(0xFFFFFFFFu, l_dot   [a], off);
            float    ob2 = __shfl_down_sync(0xFFFFFFFFu, l_score2[a], off);
            uint32_t oi2 = __shfl_down_sync(0xFFFFFFFFu, l_idx2  [a], off);
            if (ob1 > l_score[a]) {
                l_score2[a] = l_score[a]; l_idx2[a] = l_idx[a];
                l_score[a]  = ob1;        l_idx[a]  = oi1; l_dot[a] = od1;
            } else if (ob1 > l_score2[a]) {
                l_score2[a] = ob1;        l_idx2[a] = oi1;
            }
            if (ob2 > l_score2[a]) {
                l_score2[a] = ob2;        l_idx2[a] = oi2;
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            s_score [a][warp] = l_score [a];  s_idx [a][warp] = l_idx [a];  s_dot[a][warp] = l_dot[a];
            s_score2[a][warp] = l_score2[a];  s_idx2[a][warp] = l_idx2[a];
        }
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (uint32_t a = 0; a < TA32; ++a) {
            if (a >= n_live) continue;
            float    b1 = s_score [a][0];  uint32_t i1 = s_idx [a][0];  float d1 = s_dot[a][0];
            float    b2 = s_score2[a][0];  uint32_t i2 = s_idx2[a][0];
            for (uint32_t w = 1; w < TA32_WARPS; ++w) {
                float ob1 = s_score [a][w]; uint32_t oi1 = s_idx [a][w]; float od1 = s_dot[a][w];
                float ob2 = s_score2[a][w]; uint32_t oi2 = s_idx2[a][w];
                if (ob1 > b1) { b2 = b1; i2 = i1; b1 = ob1; i1 = oi1; d1 = od1; }
                else if (ob1 > b2) { b2 = ob1; i2 = oi1; }
                if (ob2 > b2) { b2 = ob2; i2 = oi2; }
            }
            uint32_t art = s_arts[a];
            d_bmu1[art]     = i1;
            d_bmu2[art]     = i2;
            d_bmu_dist[art] = sqrtf(fmaxf(0.0f, d_xnorm2[art] - 2.0f * d1 + d_norms[i1]));
        }
    }
}

// ── [INCOMPLETE] Binary BMU with L2 prefetch ("tma" variant) ────────────────
// Intended: tile-32 with software L2 prefetch of the next feature's codebook
// row to hide miss latency for rare features. Not yet benchmarked on Hopper.
//
// static __global__ void bmu_prefetch_kernel(
//     const uint32_t* __restrict__ d_row_ptr,
//     const uint16_t* __restrict__ d_col_idx,
//     const __half*   __restrict__ d_W_fm,
//     const float*    __restrict__ d_norms,
//     uint32_t* __restrict__ d_bmu1,
//     uint32_t* __restrict__ d_bmu2,
//     float*    __restrict__ d_bmu_dist,
//     uint32_t  n_samples,
//     uint32_t  N);

// ── launch helper: tile-32 binary/float with optional cluster config ─────────
static void launch_tile32(const DeviceDataset& dds, FeatureMajorCodebook& state,
                          uint32_t n_samples, cudaStream_t stream) {
    const uint32_t grid = (n_samples + TA32 - 1) / TA32;
    if (dds.is_float()) {
        static bool float_ta32_configured = false;
        constexpr size_t float_ta32_smem = TA32_MAX_DIST * TA32 * sizeof(float);
        if (!float_ta32_configured) {
            cudaFuncSetAttribute(bmu_spmm_float_kernel_ta32,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)float_ta32_smem);
            float_ta32_configured = true;
        }
        bmu_spmm_float_kernel_ta32<<<grid, TA32_BLK, float_ta32_smem, stream>>>(
            dds.d_row_ptr, dds.d_col_idx, dds.d_values, dds.d_xnorm2,
            state.d_W_fm, state.d_norms,
            state.d_bmu1, state.d_bmu2, state.d_bmu_dist, n_samples, state.n_neurons);
    } else {
        bmu_spmm_kernel_ta32<<<grid, TA32_BLK, 0, stream>>>(
            dds.d_row_ptr, dds.d_col_idx, state.d_W_fm, state.d_norms,
            state.d_bmu1, state.d_bmu2, state.d_bmu_dist, n_samples, state.n_neurons);
    }
}

void launch_bmu(const DeviceDataset& dds, FeatureMajorCodebook& state, cudaStream_t stream) {
    const uint32_t n_samples = state.n_samples;
    if (n_samples == 0) return;

    const BmuKernel variant = state.cfg.bmu_kernel;

    // ── Tile-32: larger tiles, more codebook reuse ──────────────────────────
    if (variant == BmuKernel::Tile32) {
        launch_tile32(dds, state, n_samples, stream);
        return;
    }

    // ── [INCOMPLETE] Cluster / TMA: fall back to tile32 with a warning ────
    if (variant == BmuKernel::Cluster || variant == BmuKernel::Tma) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "warning: --bmu-kernel %s is not yet implemented; falling back to tile32\n",
                         variant == BmuKernel::Cluster ? "cluster" : "tma");
            warned = true;
        }
        launch_tile32(dds, state, n_samples, stream);
        return;
    }

    // ── Default: tile-16 baseline ───────────────────────────────────────────
    const uint32_t grid = (n_samples + SPMM_TA - 1) / SPMM_TA;
    if (dds.is_float()) {
        bmu_spmm_float_kernel<<<grid, SPMM_BLK, 0, stream>>>(
            dds.d_row_ptr, dds.d_col_idx, dds.d_values, dds.d_xnorm2,
            state.d_W_fm, state.d_norms,
            state.d_bmu1, state.d_bmu2, state.d_bmu_dist, n_samples, state.n_neurons);
    } else {
        bmu_spmm_kernel<<<grid, SPMM_BLK, 0, stream>>>(
            dds.d_row_ptr, dds.d_col_idx, state.d_W_fm, state.d_norms,
            state.d_bmu1, state.d_bmu2, state.d_bmu_dist, n_samples, state.n_neurons);
    }
}

} // namespace sparsesom
