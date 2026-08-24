// som_inspect.cpp — inspect a saved .somw codebook (host-only, no GPU).
//
//   som_inspect <weights.somw>
//
// Prints the map geometry and summary statistics: overall weight range/mean, and per-neuron
// L2 norms (with a count of near-zero "dead" neurons that won nothing during training).

#include "sparsesom/codebook_io.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sparsesom;

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::printf("usage: %s <weights.somw>\n", argv[0]);
        return argc != 2 ? 2 : 0;
    }

    try {
        CodebookFile c = load_codebook(argv[1]);
        const uint32_t K = c.n_neurons();    // neurons
        const uint32_t V = c.n_features;         // features
        std::printf("codebook: %s\n", argv[1]);
        std::printf("  map: %ux%u = %u neurons,  %u features,  %zu weights\n",
                    c.map_rows, c.map_cols, K, V, c.W.size());
        if (c.W.empty() || K == 0) { std::printf("  (empty codebook)\n"); return 0; }

        // Overall weight statistics.
        double wmin = c.W[0], wmax = c.W[0], wsum = 0.0;
        for (float w : c.W) { wmin = std::fmin(wmin, w); wmax = std::fmax(wmax, w); wsum += w; }
        std::printf("  weight: min=%.4g  max=%.4g  mean=%.4g\n",
                    wmin, wmax, wsum / (double)c.W.size());

        // Per-neuron L2 norm = sqrt(sum_v W[v*K + k]^2). Feature-major layout means neuron k's
        // weights are strided by K, so accumulate across features for every neuron at once.
        std::vector<double> norm2((size_t)K, 0.0);
        for (uint32_t v = 0; v < V; ++v) {
            const float* row = &c.W[(size_t)v * K];
            for (uint32_t k = 0; k < K; ++k) { double w = row[k]; norm2[k] += w * w; }
        }
        double nmin = 1e300, nmax = 0.0, nsum = 0.0;
        uint32_t dead = 0;
        for (uint32_t k = 0; k < K; ++k) {
            double n = std::sqrt(norm2[k]);
            nmin = std::fmin(nmin, n); nmax = std::fmax(nmax, n); nsum += n;
            if (n < 1e-6) ++dead;
        }
        std::printf("  neuron L2 norm: min=%.4g  max=%.4g  mean=%.4g\n",
                    nmin, nmax, nsum / (double)K);
        std::printf("  dead neurons (norm < 1e-6): %u / %u (%.1f%%)\n",
                    dead, K, 100.0 * (double)dead / (double)K);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
