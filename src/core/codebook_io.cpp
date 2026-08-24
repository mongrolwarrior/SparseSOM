// codebook_io.cpp — .somw codebook (weights) load/save.

#include "sparsesom/codebook_io.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sparsesom {

namespace {
constexpr char kMagic[8]       = {'S', 'S', 'O', 'M', 'W', '1', '\0', '\0'};   // .ssomw
constexpr char kMagicLegacy[8] = {'S', 'B', 'S', 'O', 'M', 'W', '1', '\0'};   // .somw (back-compat)
struct Header {
    char     magic[8];
    uint32_t map_rows;
    uint32_t map_cols;
    uint32_t n_features;
    uint32_t reserved;     // always 0
};
static_assert(sizeof(Header) == 24, "Header must be tightly packed at 24 bytes");
} // namespace

void save_codebook(const std::string& path, uint32_t map_rows, uint32_t map_cols,
                   uint32_t n_features, const float* W) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_codebook: cannot open " + path);

    Header h{};
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.map_rows = map_rows;
    h.map_cols = map_cols;
    h.n_features  = n_features;
    h.reserved = 0;

    const size_t count = (size_t)n_features * map_rows * map_cols;   // V * K weights
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(W), (std::streamsize)(count * sizeof(float)));
    if (!f) throw std::runtime_error("save_codebook: write failed for " + path);
}

CodebookFile load_codebook(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_codebook: cannot open " + path);

    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) throw std::runtime_error("load_codebook: truncated header in " + path);
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 &&
        std::memcmp(h.magic, kMagicLegacy, sizeof(kMagicLegacy)) != 0)
        throw std::runtime_error("load_codebook: bad magic (not a .ssomw/.somw file): " + path);

    CodebookFile c;
    c.map_rows = h.map_rows;
    c.map_cols = h.map_cols;
    c.n_features  = h.n_features;
    c.W.resize((size_t)h.n_features * h.map_rows * h.map_cols);
    f.read(reinterpret_cast<char*>(c.W.data()), (std::streamsize)(c.W.size() * sizeof(float)));
    if (!f) throw std::runtime_error("load_codebook: truncated weights in " + path);
    return c;
}

} // namespace sparsesom
