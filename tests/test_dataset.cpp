// test_dataset.cpp — host CsrDataset: construction, .sbcsr round-trip, filtering, errors.

#include "sparsesom/dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sparsesom;

static int fails = 0;
static void check(bool ok, const char* msg) { if (!ok) { std::printf("FAIL: %s\n", msg); ++fails; } }

// Compare a dataset against expected CSR contents.
static void expect(const CsrDataset& d, uint32_t n_samples, uint32_t n_voc,
                   const std::vector<uint32_t>& rp, const std::vector<uint16_t>& ci,
                   const char* tag) {
    check(d.n_samples() == n_samples, tag);
    check(d.n_features() == n_voc, tag);
    check(d.n_nonzeros() == ci.size(), tag);
    bool rp_ok = true;
    for (uint32_t i = 0; i < rp.size() && i <= d.n_samples(); ++i) rp_ok &= (d.row_ptr()[i] == rp[i]);
    check(rp_ok, tag);
    bool ci_ok = true;
    for (uint32_t i = 0; i < ci.size() && i < d.n_nonzeros(); ++i) ci_ok &= (d.col_idx()[i] == ci[i]);
    check(ci_ok, tag);
}

int main() {
    const uint32_t V = 8;
    // 3 samples: {0,1}, {2,3}, {4}
    std::vector<uint32_t> rp = {0, 2, 4, 5};
    std::vector<uint16_t> ci = {0, 1, 2, 3, 4};

    // 1) In-memory construction.
    CsrDataset ds(rp.data(), ci.data(), 3, V);
    expect(ds, 3, V, rp, ci, "in-memory construction");

    // 2) Round-trip through the .sbcsr binary format.
    const std::string path = "/tmp/sparsesom_dataset_test.sbcsr";
    ds.save(path);
    CsrDataset ld = CsrDataset::load(path);
    expect(ld, 3, V, rp, ci, "save/load round-trip");

    // 3) filter_min_nnz drops the single-feature sample {4}.
    CsrDataset f(rp.data(), ci.data(), 3, V);
    f.filter_min_nnz(2);
    expect(f, 2, V, {0, 2, 4}, {0, 1, 2, 3}, "filter_min_nnz");

    // 4) Errors: missing file and a non-.sbcsr file must throw.
    bool threw = false;
    try { CsrDataset::load("/tmp/sparsesom_does_not_exist_zzz.sbcsr"); } catch (...) { threw = true; }
    check(threw, "load missing file should throw");

    threw = false;
    { std::FILE* g = std::fopen("/tmp/sparsesom_bad.sbcsr", "wb"); std::fputs("not an sbcsr file", g); std::fclose(g); }
    try { CsrDataset::load("/tmp/sparsesom_bad.sbcsr"); } catch (...) { threw = true; }
    check(threw, "load bad-magic file should throw");

    if (fails) { std::printf("dataset test FAILED (%d)\n", fails); return 1; }
    std::printf("dataset test OK\n");
    return 0;
}
