#include "sparsesom/featuremajor.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace sparsesom {

__global__ void fill(float* p, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

float kernel_self_test() {
  const int n = 1 << 20;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *x = nullptr, *y = nullptr;
  cudaMalloc(&x, bytes);
  cudaMalloc(&y, bytes);
  const int t = 256, b = (n + t - 1) / t;
  fill<<<b, t>>>(x, 1.0f, n);
  fill<<<b, t>>>(y, 2.0f, n);
  saxpy<<<b, t>>>(3.0f, x, y, n);          // expect y = 3*1 + 2 = 5
  cudaDeviceSynchronize();
  float y0 = 0.0f;
  cudaMemcpy(&y0, y, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(x); cudaFree(y);
  return y0;
}

const char* device_name() {
  static char name[300] = "unknown";
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) == cudaSuccess)
    std::snprintf(name, sizeof(name), "%s (sm_%d%d)", p.name, p.major, p.minor);
  return name;
}

}  // namespace sparsesom