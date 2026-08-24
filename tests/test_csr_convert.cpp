// test_csr_convert.cpp — generic CSR (.csr) ingestion: binarisation, positive-element
// selection, type narrowing, and the min_nnz row filter.

#include "sparsesom/dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace sparsesom;

static int fails = 0;
static void check(bool ok, const char* msg) { if (!ok) { std::printf("FAIL: %s\n", msg); ++fails; } }

// Write a .csr file matching CsrDataset::from_csr_file's expected layout.
static void write_csr(const std::string& path, uint32_t n_rows, uint32_t n_cols,
                      const std::vector<int32_t>& off, const std::vector<int32_t>& col,
                      const std::vector<float>& val /* empty => no values */) {
    std::ofstream f(path, std::ios::binary);
    char magic[8] = {'C', 'S', 'R', '1', 0, 0, 0, 0};
    uint64_t nnz = col.size();
    uint32_t has_values = val.empty() ? 0u : 1u, reserved = 0;
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&n_rows), 4);
    f.write(reinterpret_cast<const char*>(&n_cols), 4);
    f.write(reinterpret_cast<const char*>(&nnz), 8);
    f.write(reinterpret_cast<const char*>(&has_values), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    f.write(reinterpret_cast<const char*>(off.data()), (std::streamsize)(off.size() * 4));
    f.write(reinterpret_cast<const char*>(col.data()), (std::streamsize)(col.size() * 4));
    if (!val.empty()) f.write(reinterpret_cast<const char*>(val.data()), (std::streamsize)(val.size() * 4));
}

int main() {
    const std::string path = "/tmp/sparsesom_test_input.csr";
    const uint32_t n_cols = 8;
    // 4 rows. row1 has 2 features; row2 has 6 stored but two are non-positive (0 and -1).
    //   row0: cols 0..4, all +1                  -> 5 positive
    //   row1: cols 0,1, all +1                    -> 2 positive
    //   row2: cols 0,1,2,3,4,5 vals 1,1,0,-1,1,1  -> 4 positive (drop the 0 and -1)
    //   row3: cols 0..6, all +1                    -> 7 positive
    std::vector<int32_t> off = {0, 5, 7, 13, 20};
    std::vector<int32_t> col = {0, 1, 2, 3, 4,  0, 1,  0, 1, 2, 3, 4, 5,  0, 1, 2, 3, 4, 5, 6};
    std::vector<float>   val = {1, 1, 1, 1, 1,  1, 1,  1, 1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    write_csr(path, 4, n_cols, off, col, val);

    // 1) Ingest (binarise + positive selection), no filtering yet.
    CsrDataset d = CsrDataset::from_csr_file(path);
    check(d.n_samples() == 4, "ingest: n_samples");
    check(d.n_features() == n_cols, "ingest: n_features");
    // row2 should have kept only its 4 positive features {0,1,4,5}.
    check(d.n_nonzeros() == 5 + 2 + 4 + 7, "ingest: positive-only nnz");
    check(d.row_ptr()[2] == 7 && d.row_ptr()[3] == 11, "ingest: row2 dropped non-positive entries");
    check(d.col_idx()[7] == 0 && d.col_idx()[8] == 1 && d.col_idx()[9] == 4 && d.col_idx()[10] == 5,
          "ingest: row2 kept {0,1,4,5}");

    // 2) Filter rows with < 5 positive elements: keeps row0 (5) and row3 (7).
    d.filter_min_nnz(5);
    check(d.n_samples() == 2, "filter: rows kept");
    check(d.n_nonzeros() == 5 + 7, "filter: nnz kept");
    check(d.row_ptr()[1] == 5 && d.row_ptr()[2] == 12, "filter: row offsets");

    // 3) Without a values array, all stored entries count as present.
    write_csr(path, 4, n_cols, off, col, {});
    CsrDataset b = CsrDataset::from_csr_file(path);
    check(b.n_nonzeros() == col.size(), "no-values: all entries kept");

    // 4) Errors: too many columns for uint16, and bad magic.
    bool threw = false;
    write_csr("/tmp/sparsesom_wide.csr", 1, 70000, {0, 0}, {}, {});
    try { CsrDataset::from_csr_file("/tmp/sparsesom_wide.csr"); } catch (...) { threw = true; }
    check(threw, "n_cols > 65536 should throw");

    threw = false;
    { std::ofstream g("/tmp/sparsesom_notcsr.csr", std::ios::binary); g << "nope"; }
    try { CsrDataset::from_csr_file("/tmp/sparsesom_notcsr.csr"); } catch (...) { threw = true; }
    check(threw, "bad magic should throw");

    if (fails) { std::printf("csr_convert test FAILED (%d)\n", fails); return 1; }
    std::printf("csr_convert test OK\n");
    return 0;
}
