// test_diagnostics.cpp — topographic_error and dead_fraction against hand-computed values.
//
// 3×3 map (grid index = r*3 + c). These diagnostics read only bmu1/bmu2 and the map geometry
// (not the weights), so we set the BMU arrays directly.

#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/training.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace sparsesom;

int main() {
    SomConfig cfg;
    cfg.map_rows = 3; cfg.map_cols = 3;      // 9 neurons
    cfg.n_features  = 1;
    const uint32_t A = 4;

    FeatureMajorCodebook cb(cfg, A);

    // bmu1/bmu2 per sample (Chebyshev distance on the 3×3 grid):
    //   s0: 0(0,0) & 1(0,1) -> Cheb 1  (adjacent)      -> not a TE
    //   s1: 0(0,0) & 5(1,2) -> Cheb 2                   -> TE
    //   s2: 4(1,1) & 0(0,0) -> Cheb 1  (diagonal adj.)  -> not a TE
    //   s3: 8(2,2) & 0(0,0) -> Cheb 2                   -> TE
    // TE = 2/4 = 0.5.  bmu1 wins: {0,0,4,8} -> 3 distinct neurons used, 6 of 9 dead -> 6/9.
    std::vector<uint32_t> bmu1 = {0, 0, 4, 8}, bmu2 = {1, 5, 0, 0};
    cudaMemcpy(cb.d_bmu1, bmu1.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cb.d_bmu2, bmu2.data(), A * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int fails = 0;
    double te = topographic_error(cb);
    double df = dead_fraction(cb);
    if (std::fabs(te - 0.5) > 1e-6)            { std::printf("FAIL: TE got %f, expected 0.5\n", te); ++fails; }
    if (std::fabs(df - 6.0 / 9.0) > 1e-6)      { std::printf("FAIL: dead got %f, expected %f\n", df, 6.0/9.0); ++fails; }

    if (fails) { std::printf("diagnostics test FAILED (%d)\n", fails); return 1; }
    std::printf("diagnostics test OK (TE=%.3f, dead=%.3f)\n", te, df);
    return 0;
}
