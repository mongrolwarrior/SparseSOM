#pragma once
// device_dataset.hpp — GPU-side sparse CSR buffers (uint32 row_ptr, uint16 col_idx).
//
// Two modes, selected purely by whether a value array is attached:
//   BINARY (default) — samples are presence sets: col_idx lists the features present in each
//                      sample and there is no value array (every present feature has weight 1).
//                      ‖x‖² = nnz. d_values / d_xnorm2 stay null.
//   FLOAT  (--float) — each present feature also carries a float weight (e.g. tf-idf / counts)
//                      in d_values[nnz], and d_xnorm2[sample] = Σ x_v² is precomputed on upload.
// The BMU/update launchers dispatch on d_values: null ⇒ the binary gather-sum path, non-null
// ⇒ the float gather-multiply path. Binary (x_v ≡ 1) is the exact special case of float.
// Include from .cu only (pulls in cuda_runtime.h).

#include <cstdint>
#include <cuda_runtime.h>

namespace sparsesom {

struct DeviceDataset {
    uint32_t  n_samples = 0;
    uint32_t  n_nonzeros = 0;

    uint32_t* d_row_ptr = nullptr;  // [n_samples + 1]
    uint16_t* d_col_idx = nullptr;  // [n_nonzeros] — feature ids present in each sample

    // FLOAT mode only (both null in binary mode):
    float*    d_values  = nullptr;  // [n_nonzeros] — the weight of each present feature
    float*    d_xnorm2  = nullptr;  // [n_samples]  — Σ x_v² per sample (the BMU distance constant)

    DeviceDataset() = default;
    ~DeviceDataset();
    DeviceDataset(const DeviceDataset&)            = delete;
    DeviceDataset& operator=(const DeviceDataset&) = delete;

    // Upload the CSR structure. If h_values is non-null the dataset enters FLOAT mode: the
    // values are uploaded to d_values and d_xnorm2 is computed (Σ x_v² per sample) on the host.
    void upload(const uint32_t* h_row_ptr, const uint16_t* h_col_idx,
                uint32_t n_samples, uint32_t n_nnz, const float* h_values = nullptr);

    bool is_float() const noexcept { return d_values != nullptr; }
};

} // namespace sparsesom
