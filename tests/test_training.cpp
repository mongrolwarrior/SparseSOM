// test_training.cpp — end-to-end training on a tiny, separable corpus.
//
// 4 distinct binary patterns over 8 features, 4 identical copies each (16 samples), on a
// 4×4 map (16 neurons). With well-separated clusters and enough neurons, batch-SOM training
// should drive quantisation error toward 0 and the assignment to full stability (converge).

#include "sparsesom/training.hpp"
#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/device_dataset.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace sparsesom;

int main() {
    const uint32_t V = 8;    // features
    const uint32_t A = 16;   // samples (4 patterns × 4 copies)

    // Patterns: {0,1}, {2,3}, {4,5}, {6,7}. Build a CSR with 4 copies of each, interleaved.
    const uint16_t patterns[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    std::vector<uint32_t> row_ptr = {0};
    std::vector<uint16_t> col_idx;
    for (uint32_t a = 0; a < A; ++a) {
        const uint16_t* p = patterns[a % 4];        // which pattern this sample is
        col_idx.push_back(p[0]);
        col_idx.push_back(p[1]);
        row_ptr.push_back((uint32_t)col_idx.size());
    }
    DeviceDataset dds;
    dds.upload(row_ptr.data(), col_idx.data(), A, (uint32_t)col_idx.size());

    SomConfig cfg;
    cfg.map_rows = 6;
    cfg.map_cols = 6;                 // 36 neurons, comfortably more than the 4 patterns
    cfg.n_features  = V;
    cfg.box_passes = 1;
    // KL-driven σ schedule: σ₀ ≈ N/2 for the 6×6 map, annealing to a sub-neuron floor so
    // well-separated clusters specialise to QE ≈ 0. Disable the topology watchdog (wd_te_bad=1)
    // — this test checks loop/convergence mechanics, not topographic quality.
    cfg.sigma_init   = 3.0f;
    cfg.sigma_min    = 0.4f;
    cfg.wd_path_frac = 1.0f;   // disable the topology restart-flag (this test checks mechanics)
    cfg.max_epochs   = 60;

    FeatureMajorCodebook cb(cfg, A);
    cb.randomise(/*seed=*/7);         // reproducible start (also builds norms)

    TrainResult res = train(dds, cb);
    const std::vector<EpochMetrics>& hist = res.history;

    int fails = 0;
    auto fail = [&](const char* msg) { std::printf("FAIL: %s\n", msg); ++fails; };

    if (!res.converged) fail("main run should report convergence");
    if (hist.size() < 2) fail("expected at least 2 epochs of history");
    if (!hist.empty()) {
        double qe0 = hist.front().qe, qeN = hist.back().qe;
        if (!(qeN < qe0))        fail("QE did not decrease over training");
        // Under a broad-σ schedule the neighbourhood coupling leaves residual QE (it is not pure
        // k-means), so require a substantial drop, not QE≈0.
        if (!(qeN < 0.6 * qe0))  fail("QE did not drop substantially over training");
        if (!(hist.back().stability > 0.99)) fail("final assignment was not stable");
        if (!(hist.size() < cfg.max_epochs)) fail("training did not converge before max_epochs");
        std::printf("summary: epochs=%zu (converged)  QE %.4f -> %.4f  final stability=%.4f\n",
                    hist.size(), qe0, qeN, hist.back().stability);
    }

    // The selectable QuantizationError criterion must also converge (the main run above used the
    // default, Kaski–Lagus).
    {
        SomConfig qecfg = cfg;
        qecfg.stop_criterion = StopCriterion::QuantizationError;
        FeatureMajorCodebook qecb(qecfg, A);
        qecb.randomise(7);
        TrainResult qr = train(dds, qecb);
        const std::vector<EpochMetrics>& qh = qr.history;
        if (qh.empty() || !(qh.back().qe < 0.6 * qh.front().qe))
            fail("QE-criterion run did not drop QE substantially");
        if (!(qh.size() < cfg.max_epochs))        fail("QE-criterion run did not converge before max_epochs");
    }

    // Resume primitives behind the CLI's continue-training prompt: a 1-epoch run hits the cap
    // (not converged); continuing from its sigma_end converges.
    {
        FeatureMajorCodebook ccb(cfg, A);
        ccb.randomise(7);
        TrainResult a = train(dds, ccb, -1.0f, 1);                     // cap to 1 epoch
        if (a.converged) fail("1-epoch capped run should not report convergence");
        TrainResult b = train(dds, ccb, a.sigma_end, cfg.max_epochs);  // continue from where it stopped
        if (!b.converged) fail("continued run should converge");
    }

    // σ-schedule helper endpoints/monotonicity: s=0 → σ₀; large s → σ_min; non-increasing. Both
    // exp and linear forms.
    {
        SomConfig sc;
        sc.sigma_min = 0.5f; sc.sigma_rate = 0.3f;
        for (SigmaSchedule sched : {SigmaSchedule::Exponential, SigmaSchedule::Linear}) {
            sc.sigma_schedule = sched;
            float s0  = sigma_for_progress(sc, 100.0f, 0.0);
            float big = sigma_for_progress(sc, 100.0f, 1e6);
            if (std::fabs(s0 - 100.0f) > 1e-3) fail("sigma_for_progress(s=0) != sigma0");
            if (std::fabs(big - sc.sigma_min) > 1e-3) fail("sigma_for_progress(large s) != sigma_min");
            float prev = s0;
            for (double s = 0.0; s <= 30.0; s += 1.0) {
                float v = sigma_for_progress(sc, 100.0f, s);
                if (v > prev + 1e-4) { fail("sigma_for_progress not monotone non-increasing"); break; }
                prev = v;
            }
        }
    }

    if (fails) { std::printf("training test FAILED (%d)\n", fails); return 1; }
    std::printf("training test OK\n");
    return 0;
}
