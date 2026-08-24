#pragma once
// pca_init.hpp — initialize a feature-major codebook from precomputed PCA components.
//
// Reads a .sompca file (mean + top-k principal component directions + singular values,
// ~360 KB for k=2, V=30766) and generates the full codebook grid directly in GPU memory.
// No host-side codebook buffer, no disk write — works at any map size.

#include <cstdint>
#include <string>

namespace sparsesom {

struct FeatureMajorCodebook;

// Initialize cb's d_W_fm from PCA components in sompca_path.  Each neuron at grid
// position (r, c) gets:
//   w[v] = mean[v] + scale0 * r_norm * pc0[v] + scale1 * c_norm * pc1[v]
// where r_norm, c_norm ∈ [-1, 1] span the map dimensions and scale_i = sv_i / √n_samples.
// Writes directly to d_W_fm as __half; calls recompute_norms() afterwards.
void pca_init(FeatureMajorCodebook& cb, const std::string& sompca_path,
              uint32_t n_train_samples);

} // namespace sparsesom
