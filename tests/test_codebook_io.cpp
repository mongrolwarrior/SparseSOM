// test_codebook_io.cpp — .somw codebook save/load round-trip and error handling (host-only).

#include "sparsesom/codebook_io.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace sparsesom;

static int fails = 0;
static void check(bool ok, const char* msg) { if (!ok) { std::printf("FAIL: %s\n", msg); ++fails; } }

int main() {
    const uint32_t rows = 2, cols = 3, V = 4;     // 6 neurons, 4 features
    const uint32_t K = rows * cols;
    std::vector<float> W((size_t)V * K);
    for (size_t i = 0; i < W.size(); ++i) W[i] = 0.5f * (float)i;   // distinct values

    const std::string path = "/tmp/sparsesom_test.somw";
    save_codebook(path, rows, cols, V, W.data());

    CodebookFile c = load_codebook(path);
    check(c.map_rows == rows && c.map_cols == cols, "round-trip: geometry");
    check(c.n_features == V && c.n_neurons() == K, "round-trip: feature/neuron counts");
    check(c.W.size() == W.size(), "round-trip: weight count");
    bool same = (c.W.size() == W.size());
    for (size_t i = 0; same && i < W.size(); ++i) same &= (c.W[i] == W[i]);
    check(same, "round-trip: weights identical");

    // Errors: missing file and bad magic.
    bool threw = false;
    try { load_codebook("/tmp/sparsesom_no_such.somw"); } catch (...) { threw = true; }
    check(threw, "missing file should throw");

    threw = false;
    { std::ofstream g("/tmp/sparsesom_bad.somw", std::ios::binary); g << "not a somw file...."; }
    try { load_codebook("/tmp/sparsesom_bad.somw"); } catch (...) { threw = true; }
    check(threw, "bad magic should throw");

    if (fails) { std::printf("codebook_io test FAILED (%d)\n", fails); return 1; }
    std::printf("codebook_io test OK\n");
    return 0;
}
