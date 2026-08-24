#pragma once
// codebook_io.hpp — save/load a trained feature-major codebook to the .somw binary format
// (host-only, no CUDA). Layout: a 24-byte header
//   { char magic[8]="SBSOMW1"; uint32 map_rows; uint32 map_cols; uint32 n_features; uint32 reserved; }
// followed by n_features * (map_rows*map_cols) float32 weights in feature-major order
// (W[v*K + k] for feature v, neuron k). The GPU side downloads d_W_fm into a host buffer
// and calls save_codebook(); load_codebook() reads it back for inspection without a GPU.

#include <cstdint>
#include <string>
#include <vector>

namespace sparsesom {

// In-memory contents of a loaded .somw file.
struct CodebookFile {
    uint32_t           map_rows = 0;
    uint32_t           map_cols = 0;
    uint32_t           n_features  = 0;
    std::vector<float> W;                       // [n_features * n_neurons], feature-major W[v*K + k]
    uint32_t n_neurons() const { return map_rows * map_cols; }
};

// Write the feature-major codebook. W must point to n_features * (map_rows*map_cols) floats.
void save_codebook(const std::string& path, uint32_t map_rows, uint32_t map_cols,
                   uint32_t n_features, const float* W);

// Read a .somw file. Throws std::runtime_error on a missing file, bad magic, or a size
// that doesn't match the header's dimensions.
CodebookFile load_codebook(const std::string& path);

} // namespace sparsesom
