// device_dataset.cu — upload a sparse CSR corpus (binary or float) to the GPU.

#include "sparsesom/device_dataset.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace sparsesom {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA: ") +                    \
                                     cudaGetErrorString(_e) + " [" #expr "]");  \
    } while (0)

DeviceDataset::~DeviceDataset() {
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_xnorm2);
}

void DeviceDataset::upload(const uint32_t* h_row_ptr, const uint16_t* h_col_idx,
                           uint32_t n_samp, uint32_t n_nnz, const float* h_values) {
    n_samples  = n_samp;
    n_nonzeros = n_nnz;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n_samples + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)n_nnz * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (size_t)(n_samples + 1) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, (size_t)n_nnz * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    if (h_values) {                                  // FLOAT mode: upload values + precompute ‖x‖²
        CUDA_CHECK(cudaMalloc(&d_values, (size_t)n_nnz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_values, h_values, (size_t)n_nnz * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::vector<float> xn2(n_samples);           // Σ x_v² per sample (the BMU ‖x‖² constant)
        for (uint32_t a = 0; a < n_samples; ++a) {
            double s = 0.0;
            for (uint32_t j = h_row_ptr[a]; j < h_row_ptr[a + 1]; ++j)
                s += (double)h_values[j] * (double)h_values[j];
            xn2[a] = (float)s;
        }
        CUDA_CHECK(cudaMalloc(&d_xnorm2, (size_t)n_samples * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_xnorm2, xn2.data(), (size_t)n_samples * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

} // namespace sparsesom
