// dataset.cpp — host CSR corpus: in-memory construction, .sbcsr load/save, filtering.

#include "sparsesom/dataset.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sparsesom {

// On-disk header for the .sbcsr format: a magic tag, the three sizes, and padding to keep
// the following arrays 8-byte aligned. The row_ptr (uint32 × n_samples+1) and col_idx
// (uint16 × n_nonzeros) arrays follow immediately, little-endian.
namespace {
constexpr char kMagic[8]       = {'S', 'C', 'S', 'R', '1', '\0', '\0', '\0'};   // .scsr
constexpr char kMagicLegacy[8] = {'S', 'B', 'C', 'S', 'R', '1', '\0', '\0'};   // .sbcsr (back-compat)
struct Header {
    char     magic[8];
    uint32_t n_samples;
    uint32_t n_features;
    uint32_t n_nonzeros;
    uint32_t has_values;   // 1 ⇒ a float32 values[n_nonzeros] block follows col_idx; 0 ⇒ binary.
                           // Legacy .sbcsr left this field 0 ("reserved"), so it reads back as a
                           // binary corpus — the back-compatible no-values special case.
};
static_assert(sizeof(Header) == 24, "Header must be tightly packed at 24 bytes");
} // namespace

CsrDataset::CsrDataset(const uint32_t* row_ptr, const uint16_t* col_idx,
                       uint32_t n_samples, uint32_t n_features)
    : n_samples_(n_samples), n_features_(n_features),
      row_ptr_(row_ptr, row_ptr + n_samples + 1),
      col_idx_(col_idx, col_idx + row_ptr[n_samples]) {}

void CsrDataset::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("CsrDataset::save: cannot open " + path);

    Header h{};
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.n_samples = n_samples_;
    h.n_features    = n_features_;
    h.n_nonzeros = (uint32_t)col_idx_.size();
    h.has_values = values_.empty() ? 0u : 1u;

    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(row_ptr_.data()),
            (std::streamsize)(row_ptr_.size() * sizeof(uint32_t)));
    f.write(reinterpret_cast<const char*>(col_idx_.data()),
            (std::streamsize)(col_idx_.size() * sizeof(uint16_t)));
    if (h.has_values)
        f.write(reinterpret_cast<const char*>(values_.data()),
                (std::streamsize)(values_.size() * sizeof(float)));
    if (!f) throw std::runtime_error("CsrDataset::save: write failed for " + path);
}

CsrDataset CsrDataset::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("CsrDataset::load: cannot open " + path);

    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) throw std::runtime_error("CsrDataset::load: truncated header in " + path);
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 &&
        std::memcmp(h.magic, kMagicLegacy, sizeof(kMagicLegacy)) != 0)
        throw std::runtime_error("CsrDataset::load: bad magic (not a .scsr/.sbcsr file): " + path);

    CsrDataset d;
    d.n_samples_ = h.n_samples;
    d.n_features_    = h.n_features;
    d.row_ptr_.resize((size_t)h.n_samples + 1);
    d.col_idx_.resize(h.n_nonzeros);
    f.read(reinterpret_cast<char*>(d.row_ptr_.data()),
           (std::streamsize)(d.row_ptr_.size() * sizeof(uint32_t)));
    f.read(reinterpret_cast<char*>(d.col_idx_.data()),
           (std::streamsize)(d.col_idx_.size() * sizeof(uint16_t)));
    if (h.has_values) {                                   // float corpus: read the values block
        d.values_.resize(h.n_nonzeros);
        f.read(reinterpret_cast<char*>(d.values_.data()),
               (std::streamsize)(d.values_.size() * sizeof(float)));
    }
    if (!f) throw std::runtime_error("CsrDataset::load: truncated data in " + path);

    // Cheap sanity checks (O(1)): the CSR must start at 0 and end at n_nonzeros.
    if (d.row_ptr_.front() != 0 || d.row_ptr_.back() != h.n_nonzeros)
        throw std::runtime_error("CsrDataset::load: inconsistent row_ptr in " + path);
    return d;
}

// Header for the generic/standard CSR input format (.csr). Laid out so the fields are
// naturally aligned with no padding (uint64 nnz sits at offset 16).
namespace {
constexpr char kCsrMagic[8] = {'C', 'S', 'R', '1', '\0', '\0', '\0', '\0'};
struct CsrInHeader {
    char     magic[8];
    uint32_t n_rows;
    uint32_t n_cols;
    uint64_t nnz;
    uint32_t has_values;   // 1 if a float32 values array follows col_indices
    uint32_t reserved;
};
static_assert(sizeof(CsrInHeader) == 32, "CsrInHeader must be 32 bytes with no padding");
} // namespace

CsrDataset CsrDataset::from_csr_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("from_csr_file: cannot open " + path);

    CsrInHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) throw std::runtime_error("from_csr_file: truncated header in " + path);
    if (std::memcmp(h.magic, kCsrMagic, sizeof(kCsrMagic)) != 0)
        throw std::runtime_error("from_csr_file: bad magic (not a .csr file): " + path);
    if (h.n_cols > 65536)   // col indices must fit uint16 (values are in [0, n_cols))
        throw std::runtime_error("from_csr_file: n_cols > 65536, too many features for uint16");

    // Read the standard-CSR arrays (int32 offsets/indices, optional float values).
    std::vector<int32_t> off(h.n_rows + 1);
    std::vector<int32_t> col(h.nnz);
    std::vector<float>   val;
    f.read(reinterpret_cast<char*>(off.data()), (std::streamsize)(off.size() * sizeof(int32_t)));
    f.read(reinterpret_cast<char*>(col.data()), (std::streamsize)(col.size() * sizeof(int32_t)));
    if (h.has_values) {
        val.resize(h.nnz);
        f.read(reinterpret_cast<char*>(val.data()), (std::streamsize)(val.size() * sizeof(float)));
    }
    if (!f) throw std::runtime_error("from_csr_file: truncated data in " + path);
    if ((uint64_t)off.front() != 0 || (uint64_t)off.back() != h.nnz)
        throw std::runtime_error("from_csr_file: inconsistent row_offsets in " + path);

    // Binarise into the .sbcsr layout: each stored entry is a present feature; when values
    // exist, keep only strictly-positive ones. Narrow indices to uint32/uint16.
    CsrDataset d;
    d.n_samples_ = h.n_rows;
    d.n_features_    = h.n_cols;
    d.row_ptr_.reserve((size_t)h.n_rows + 1);
    d.col_idx_.reserve(h.nnz);
    d.row_ptr_.push_back(0);
    for (uint32_t i = 0; i < h.n_rows; ++i) {
        for (int32_t k = off[i]; k < off[i + 1]; ++k) {
            if (h.has_values && !(val[k] > 0.0f)) continue;     // keep positive entries only
            int32_t c = col[k];
            if (c < 0 || (uint32_t)c >= h.n_cols)
                throw std::runtime_error("from_csr_file: column index out of range in " + path);
            d.col_idx_.push_back((uint16_t)c);
        }
        d.row_ptr_.push_back((uint32_t)d.col_idx_.size());
    }
    return d;
}

void CsrDataset::filter_min_nnz(uint32_t min_nnz) {
    std::vector<uint32_t> new_rp;
    std::vector<uint16_t> new_ci;
    new_rp.reserve(row_ptr_.size());
    new_ci.reserve(col_idx_.size());
    new_rp.push_back(0);
    for (uint32_t a = 0; a < n_samples_; ++a) {
        uint32_t s = row_ptr_[a], e = row_ptr_[a + 1];
        if (e - s >= min_nnz) {                       // keep samples with enough features
            for (uint32_t j = s; j < e; ++j) new_ci.push_back(col_idx_[j]);
            new_rp.push_back((uint32_t)new_ci.size());
        }
    }
    n_samples_ = (uint32_t)new_rp.size() - 1;
    row_ptr_ = std::move(new_rp);
    col_idx_ = std::move(new_ci);
}

} // namespace sparsesom
