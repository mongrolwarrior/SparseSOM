// csr_convert.cpp — convert a generic/standard CSR binary (.csr) into SparseSOM's
// sparse-binary .sbcsr format.
//
//   csr_convert <in.csr> <out.sbcsr> [--min-nnz N]
//
// Binarises the input (positive entries become present features), narrows the index types,
// and drops rows with fewer than --min-nnz positive features (default 5; a MEDLINE-style
// assumption that very sparse rows aren't useful). See CsrDataset::from_csr_file for the
// input layout.

#include "sparsesom/dataset.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace sparsesom;

static void usage(const char* prog) {
    std::printf("usage: %s <in.csr> <out.sbcsr> [--min-nnz N]   (default min-nnz 5)\n", prog);
}

int main(int argc, char** argv) {
    std::string in, out;
    uint32_t min_nnz = 5;   // exclude rows with fewer than this many positive elements

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--min-nnz") {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for --min-nnz\n"); return 2; }
            min_nnz = (uint32_t)std::atoi(argv[++i]);
        }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); return 2;
        }
        else if (in.empty())  in = a;
        else if (out.empty())  out = a;
        else { std::fprintf(stderr, "unexpected argument: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (in.empty() || out.empty()) { usage(argv[0]); return 2; }

    try {
        CsrDataset d = CsrDataset::from_csr_file(in);
        uint32_t rows_in = d.n_samples();
        d.filter_min_nnz(min_nnz);
        d.save(out);
        std::printf("converted %s -> %s\n"
                    "  rows: %u -> %u  (dropped %u with < %u positive features)\n"
                    "  features: %u   nonzeros: %u\n",
                    in.c_str(), out.c_str(), rows_in, d.n_samples(),
                    rows_in - d.n_samples(), min_nnz, d.n_features(), d.n_nonzeros());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
