#include "sparsesom/featuremajor.hpp"
#include <cstdio>
int main() {
  float r = sparsesom::kernel_self_test();
  if (r != 5.0f) { std::printf("smoke FAIL: got %f\n", r); return 1; }
  std::printf("smoke OK\n");
  return 0;
}