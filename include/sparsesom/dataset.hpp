#pragma once
// dataset.hpp — host-side sparse-binary CSR corpus loader (no CUDA).
//
// A corpus is a set of samples; each sample is the set of feature ids present in it (binary,
// no values). Stored as CSR: row_ptr[a]..row_ptr[a+1] indexes that sample's features in
// col_idx. This class owns its arrays and can be built three ways:
//   - from in-memory CSR arrays (copied in),
//   - loaded from the compact self-describing .sbcsr binary format (see dataset.cpp),
//   - (then optionally compacted with filter_min_nnz).
// To train, hand row_ptr()/col_idx()/n_samples()/n_nonzeros() to DeviceDataset::upload and
// set SomConfig::n_features to n_features().

#include <cstdint>
#include <string>
#include <vector>

namespace sparsesom {

class CsrDataset {
public:
    CsrDataset() = default;

    // Build from caller-owned CSR arrays (copied in).
    //   row_ptr  — [n_samples + 1]; row_ptr[0]==0 and row_ptr[n_samples]==n_nonzeros
    //   col_idx  — the present feature ids, concatenated per sample; each value must be < n_features
    CsrDataset(const uint32_t* row_ptr, const uint16_t* col_idx,
               uint32_t n_samples, uint32_t n_features);

    // Load from / save to the .sbcsr binary format. load() throws std::runtime_error on a
    // missing file, bad magic, or inconsistent sizes.
    static CsrDataset load(const std::string& path);
    void save(const std::string& path) const;

    // Ingest a generic/standard CSR binary (the ".csr" format, closest to cuSPARSE CSR) and
    // binarise it into a sparse-binary corpus. Layout: a 32-byte header
    //   { char magic[8]="CSR1"; uint32 n_rows; uint32 n_cols; uint64 nnz;
    //     uint32 has_values; uint32 reserved; }
    // then int32 row_offsets[n_rows+1], int32 col_indices[nnz], and (iff has_values)
    // float32 values[nnz]. Each stored entry becomes a present feature; when values are
    // present only strictly-positive entries are kept. Types are narrowed to the .sbcsr
    // layout (uint32 offsets, uint16 cols), so n_cols must be <= 65536. Does NOT filter by
    // row length — call filter_min_nnz() afterwards. Throws on bad magic / sizes / an
    // out-of-range column / too many columns for uint16.
    static CsrDataset from_csr_file(const std::string& path);

    // Drop samples with fewer than min_nnz features; compacts row_ptr/col_idx in place.
    void filter_min_nnz(uint32_t min_nnz);

    uint32_t n_samples()  const noexcept { return n_samples_; }
    uint32_t n_features()     const noexcept { return n_features_; }
    uint32_t n_nonzeros()  const noexcept { return (uint32_t)col_idx_.size(); }
    const uint32_t* row_ptr() const noexcept { return row_ptr_.data(); }
    const uint16_t* col_idx() const noexcept { return col_idx_.data(); }
    // Float corpus only: per-nonzero feature weights (empty for a binary corpus). When present,
    // hand to DeviceDataset::upload to drive the gather-multiply (float) BMU/update path.
    bool has_values()       const noexcept { return !values_.empty(); }
    const float* values()   const noexcept { return values_.data(); }

private:
    uint32_t              n_samples_ = 0;   // number of samples (rows)
    uint32_t              n_features_    = 0;   // feature-id space size (col_idx values are < this)
    std::vector<uint32_t> row_ptr_;          // [n_samples + 1]
    std::vector<uint16_t> col_idx_;          // [n_nonzeros]
    std::vector<float>    values_;           // [n_nonzeros] iff a float corpus; empty ⇒ binary
};

} // namespace sparsesom
