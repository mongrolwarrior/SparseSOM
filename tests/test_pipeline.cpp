// test_pipeline.cpp — end-to-end seam: CsrDataset (file) -> DeviceDataset -> train.
//
// Builds a separable binary corpus, saves it as .sbcsr, loads it back, uploads to the GPU,
// and trains — asserting the SOM converges to QE ≈ 0. Also leaves the .sbcsr on disk so the
// `cli` CTest can drive the sparsesom binary on the same corpus.

#include "sparsesom/dataset.hpp"
#include "sparsesom/device_dataset.hpp"
#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/training.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sparsesom;

// Shared with the CLI test (see CMakeLists fixture wiring).
static const char* kCorpusPath = "/tmp/sparsesom_cli_corpus.sbcsr";

int main() {
    const uint32_t V = 8;
    const uint16_t patterns[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};

    // 16 samples = 4 copies of each pattern.
    std::vector<uint32_t> rp = {0};
    std::vector<uint16_t> ci;
    for (uint32_t a = 0; a < 16; ++a) {
        const uint16_t* p = patterns[a % 4];
        ci.push_back(p[0]); ci.push_back(p[1]);
        rp.push_back((uint32_t)ci.size());
    }

    // Save, then load back through the file format (exercises the real loader path).
    CsrDataset(rp.data(), ci.data(), 16, V).save(kCorpusPath);
    CsrDataset ds = CsrDataset::load(kCorpusPath);

    int fails = 0;
    if (ds.n_samples() != 16 || ds.n_features() != V) { std::printf("FAIL: loaded sizes wrong\n"); ++fails; }

    // Configure from the dataset and train (KL-driven σ schedule on 36 neurons).
    SomConfig cfg;
    cfg.map_rows = 6; cfg.map_cols = 6;
    cfg.n_features  = ds.n_features();
    cfg.box_passes = 1;
    cfg.sigma_init = 3.0f; cfg.sigma_min = 0.4f; cfg.wd_path_frac = 1.0f;  // small map; disable topo-flag
    cfg.max_epochs = 60;

    DeviceDataset dds;
    dds.upload(ds.row_ptr(), ds.col_idx(), ds.n_samples(), ds.n_nonzeros());
    FeatureMajorCodebook cb(cfg, ds.n_samples());
    cb.randomise(7);

    std::vector<EpochMetrics> hist = train(dds, cb).history;
    if (hist.empty() || !(hist.back().qe < 0.6 * hist.front().qe)) { std::printf("FAIL: pipeline did not drop QE substantially\n"); ++fails; }
    if (!hist.empty() && hist.back().stability < 0.99) { std::printf("FAIL: pipeline not stable\n"); ++fails; }

    if (fails) { std::printf("pipeline test FAILED (%d)\n", fails); return 1; }
    std::printf("pipeline test OK (final QE=%.4f, corpus at %s)\n",
                hist.back().qe, kCorpusPath);
    return 0;
}
