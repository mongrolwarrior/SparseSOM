// main.cpp — SparseSOM command-line entry point: load a corpus, train, report.
//
//   sparsesom <corpus.scsr> [options]
//
// Loads a sparse-binary CSR corpus, builds a feature-major codebook sized from it, and runs
// the batch-SOM training loop, printing per-epoch progress and a final summary.

#include "sparsesom/dataset.hpp"
#include "sparsesom/device_dataset.hpp"
#include "sparsesom/feature_major_codebook.hpp"
#include "sparsesom/training.hpp"
#include "sparsesom/codebook_io.hpp"    // save_codebook()
#include "sparsesom/pca_init.hpp"       // pca_init()
#include "sparsesom/featuremajor.hpp"   // device_name()
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>       // isatty
#include <sys/statvfs.h>  // statvfs — disk free space

using namespace sparsesom;

static void usage(const char* prog) {
    SomConfig d;   // defaults to show
    std::printf(
        "usage: %s <corpus.scsr> [options]\n"
        "  --rows N        map rows           (default %u)\n"
        "  --cols N        map cols           (default %u)\n"
        "  --epochs N      max epochs         (default %u)\n"
        "  --sigma-init F  initial sigma0, box half-width (default %.2f; recommend ~N/2)\n"
        "  --sigma-min F   sigma floor       (default %.2f; <0.5 to collapse to the BMU)\n"
        "  --sigma-sched S exp|linear|det-exp sigma interpolation (default exp)\n"
        "  --sigma-rate F  interpolation rate over KL-progress (default %.2f)\n"
        "  --sigma-gref F  fractional-KL-improvement scale (default %.3f)\n"
        "  --sigma-window N  epochs over which KL improvement is measured (default %u)\n"
        "  --sigma-accel F watchdog rate accel on plateau-while-large (default %.2f)\n"
        "  --wd-path-frac F watchdog: restart unless KL path term falls to <= F x its peak (default %.2f)\n"
        "  --te-converge-max F  max topographic error to call a map converged (default %.2f)\n"
        "  --min-nnz N     drop samples with < N features (default 0)\n"
        "  --seed N        codebook init seed (default 42)\n"
        "  --save-weights PATH  write the trained codebook to PATH (.ssomw)\n"
        "  --load-weights PATH  start from a saved .ssomw codebook (map geometry from the file)\n"
        "  --eval               assign only and report QE; no training (use with --load-weights)\n"
        "  --bin                binary gather-sum path (default is float gather-multiply)\n"
        "  --check-fit          reserve the full training footprint, report fit/OOM, then exit\n"
        "  --eval-corpus PATH   after training, evaluate this held-out corpus and print metrics JSON\n"
        "  --init-pca PATH      initialize codebook from PCA components (.sompca) instead of random\n"
        "  --dump-bmu PATH      write the per-sample winning neuron (uint32[n]) — the clustering\n"
        "  --bmu-kernel K       BMU kernel variant: tile16 (default), tile32, cluster*, tma* (*=incomplete)\n",
        prog, d.map_rows, d.map_cols, d.max_epochs,
        (double)d.sigma_init, (double)d.sigma_min, (double)d.sigma_rate, (double)d.sigma_g_ref,
        d.sigma_window, (double)d.sigma_accel, (double)d.wd_path_frac, (double)d.te_converge_max);
}

// Ask whether to keep training and for how many more epochs. Returns the number of additional
// epochs (>0), or 0 to stop. Non-interactive (stdin is not a TTY — tests, cron, pipes) always
// returns 0, so a run that hits the epoch cap never blocks waiting for input.
static uint32_t prompt_more_epochs() {
    if (!isatty(fileno(stdin))) {
        std::printf("(non-interactive: stopping; re-run with more --epochs to train longer)\n");
        return 0;
    }
    std::printf("continue training? additional epochs [default 20, 'n' to stop]: ");
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) return 0;            // EOF → stop
    size_t i = 0; while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
    std::string s = line.substr(i);
    if (s.empty())                  return 20;              // Enter → default 20 more epochs
    if (s[0] == 'n' || s[0] == 'N') return 0;               // 'n'/'no' → stop
    int v = std::atoi(s.c_str());
    return v > 0 ? (uint32_t)v : 0;                         // a positive number, else stop
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    std::string corpus;          // path to the .scsr file
    SomConfig   cfg;             // training configuration, starts at defaults
    uint32_t    min_nnz = 0;     // drop samples with fewer than this many features
    unsigned    seed    = 42;    // codebook initialisation seed
    std::string save_weights;    // if set, write the trained codebook here (.ssomw)
    std::string load_weights;    // if set, start from this saved .ssomw instead of randomising
    bool        eval    = false; // assign only and report QE, no training
    bool        check_fit = false; // reserve the full training footprint and report fit/OOM only
    bool        use_float = true;  // default: sparse-FLOAT path (gather-multiply); --bin forces binary
    std::string dump_bmu;        // if set, write per-sample winning neuron (uint32[n]) here
    std::string eval_corpus;     // if set, evaluate this held-out corpus after training
    std::string init_pca;        // if set, initialize codebook from PCA components (.sompca)

    // Parse args. next() returns the value following a flag, or errors if absent.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if      (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--rows")       cfg.map_rows         = (uint32_t)std::atoi(next("--rows"));
        else if (a == "--cols")       cfg.map_cols         = (uint32_t)std::atoi(next("--cols"));
        else if (a == "--epochs")     cfg.max_epochs       = (uint32_t)std::atoi(next("--epochs"));
        else if (a == "--sigma-init") cfg.sigma_init  = (float)std::atof(next("--sigma-init"));
        else if (a == "--sigma-min")  cfg.sigma_min   = (float)std::atof(next("--sigma-min"));
        else if (a == "--sigma-rate") cfg.sigma_rate  = (float)std::atof(next("--sigma-rate"));
        else if (a == "--sigma-gref") cfg.sigma_g_ref = (float)std::atof(next("--sigma-gref"));
        else if (a == "--sigma-accel") cfg.sigma_accel = (float)std::atof(next("--sigma-accel"));
        else if (a == "--wd-path-frac") cfg.wd_path_frac = (float)std::atof(next("--wd-path-frac"));
        else if (a == "--sigma-window") cfg.sigma_window = (uint32_t)std::atoi(next("--sigma-window"));
        else if (a == "--sigma-sched") {
            std::string v = next("--sigma-sched");
            if      (v == "linear")  cfg.sigma_schedule = SigmaSchedule::Linear;
            else if (v == "det-exp") cfg.sigma_schedule = SigmaSchedule::DeterministicExp;
            else                     cfg.sigma_schedule = SigmaSchedule::Exponential;
        }
        else if (a == "--te-converge-max") cfg.te_converge_max = (float)std::atof(next("--te-converge-max"));
        else if (a == "--min-nnz")    min_nnz              = (uint32_t)std::atoi(next("--min-nnz"));
        else if (a == "--seed")       seed                 = (unsigned)std::atoi(next("--seed"));
        else if (a == "--save-weights") save_weights       = next("--save-weights");
        else if (a == "--load-weights") load_weights       = next("--load-weights");
        else if (a == "--eval")         eval               = true;
        else if (a == "--check-fit")    check_fit          = true;
        else if (a == "--bin")          use_float          = false;
        else if (a == "--float")        use_float          = true;   // now the default; accepted for back-compat
        else if (a == "--eval-corpus")  eval_corpus        = next("--eval-corpus");
        else if (a == "--init-pca")     init_pca           = next("--init-pca");
        else if (a == "--dump-bmu")     dump_bmu           = next("--dump-bmu");
        else if (a == "--bmu-kernel") {
            std::string v = next("--bmu-kernel");
            if      (v == "tile16")  cfg.bmu_kernel = BmuKernel::Tile16;
            else if (v == "tile32")  cfg.bmu_kernel = BmuKernel::Tile32;
            else if (v == "cluster") cfg.bmu_kernel = BmuKernel::Cluster;
            else if (v == "tma")     cfg.bmu_kernel = BmuKernel::Tma;
            else throw std::runtime_error("unknown --bmu-kernel: " + v +
                                          " (tile16|tile32|cluster|tma)");
        }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0]); return 2;
        }
        else corpus = a;         // the positional corpus path
    }
    if (corpus.empty()) { usage(argv[0]); return 2; }

    try {
        std::printf("GPU: %s\n", device_name());

        // Load the corpus; the feature dimension comes from the file.
        CsrDataset ds = CsrDataset::load(corpus);
        if (min_nnz > 0) ds.filter_min_nnz(min_nnz);
        cfg.n_features = ds.n_features();

        // --check-fit: reserve the full TRAINING footprint (dataset already on the GPU from the
        // load above + codebook + BMU arrays + the factorized-update tiles) and report fit/OOM,
        // with NO training. Lets a capacity search probe map sizes in ~a second each instead of
        // running epochs — the OOM is purely a memory question, independent of the data values.
        if (check_fit) {
            FeatureMajorCodebook cb(cfg, ds.n_samples());      // codebook + norms + BMU arrays
            const uint32_t K = cfg.n_neurons(), CH = std::min<uint32_t>(2048, cfg.n_features);
            void* t[4];
            const size_t sz[4] = {(size_t)K * CH * 4, (size_t)K * CH * 4, (size_t)K * 4, (size_t)K * 4};
            for (int i = 0; i < 4; ++i) {                      // the update tiles (S, tmp, m, tmp_m)
                cudaError_t e = cudaMalloc(&t[i], sz[i]);
                if (e != cudaSuccess)
                    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e) +
                                             " [check-fit update tile]");
            }
            for (int i = 0; i < 4; ++i) cudaFree(t[i]);
            std::printf("check-fit: FIT  %ux%u (%u neurons, %u features) — full footprint reserved\n",
                        cfg.map_rows, cfg.map_cols, K, cfg.n_features);
            return 0;
        }

        // If restoring a saved codebook, its geometry wins (map dims come from the file) and
        // its feature space must match the corpus.
        CodebookFile loaded;
        const bool have_loaded = !load_weights.empty();
        if (have_loaded) {
            loaded = load_codebook(load_weights);
            if (loaded.n_features != cfg.n_features)
                throw std::runtime_error("loaded codebook has " + std::to_string(loaded.n_features) +
                                         " features but the corpus has " + std::to_string(cfg.n_features));
            cfg.map_rows = loaded.map_rows;
            cfg.map_cols = loaded.map_cols;
        }

        std::printf("corpus: %s — %u samples, %u features, %u nonzeros\n",
                    corpus.c_str(), ds.n_samples(), ds.n_features(), ds.n_nonzeros());
        std::printf("map: %ux%u = %u neurons\n",
                    cfg.map_rows, cfg.map_cols, cfg.n_neurons());

        // Upload to the GPU and build the codebook — restored from file or randomised.
        // Default is the sparse-FLOAT path (gather-multiply, ‖x‖²=Σx_v²). If the corpus carries a
        // values block (.scsr has_values), those real weights are used; otherwise (a valueless
        // .scsr / legacy .scsr) unit values x_v≡1 are synthesized — the binary special case, which
        // matches binary output exactly. --bin selects the binary gather-sum path (no values array).
        DeviceDataset dds;
        std::vector<float> values;
        const float* hvals = nullptr;
        if (!use_float) {
            std::printf("mode: sparse-BINARY (gather-sum)\n");
        } else if (ds.has_values()) {
            hvals = ds.values();
            std::printf("mode: sparse-FLOAT (gather-multiply; %u stored values)\n", ds.n_nonzeros());
        } else {
            values.assign(ds.n_nonzeros(), 1.0f);
            hvals = values.data();
            std::printf("mode: sparse-FLOAT (gather-multiply; no stored values → x_v=1)\n");
        }
        {
            const char* knames[] = {"tile16", "tile32", "cluster", "tma"};
            std::printf("bmu-kernel: %s\n", knames[(int)cfg.bmu_kernel]);
        }
        dds.upload(ds.row_ptr(), ds.col_idx(), ds.n_samples(), ds.n_nonzeros(), hvals);
        FeatureMajorCodebook cb(cfg, ds.n_samples());
        if (have_loaded) { cb.upload_W_fm(loaded.W); std::printf("loaded codebook from %s\n", load_weights.c_str()); }
        else if (!init_pca.empty()) pca_init(cb, init_pca, ds.n_samples());
        else             cb.randomise(seed);

        // Write the per-sample BMU (the SOM clustering, uint32[n_samples]) if requested. A current
        // assignment pass must have run (eval, or post-training) so d_bmu1 is up to date.
        auto write_bmu = [&]() {
            if (dump_bmu.empty()) return;
            std::vector<uint32_t> bmu;
            cb.download_bmu1(bmu);
            FILE* f = std::fopen(dump_bmu.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot open --dump-bmu file: " + dump_bmu);
            std::fwrite(bmu.data(), sizeof(uint32_t), bmu.size(), f);
            std::fclose(f);
            std::printf("dumped %zu BMUs (uint32) to %s\n", bmu.size(), dump_bmu.c_str());
        };

        // Eval mode: assign the corpus with the current codebook and report QE, no training.
        if (eval) {
            double qe = evaluate_qe(dds, cb);
            std::printf("eval: QE=%.4f over %u samples (no training)\n", qe, ds.n_samples());
            write_bmu();
            return 0;
        }

        // Train. If a run hits the epoch cap without converging, show a summary of the map's
        // characteristics and offer to continue — the σ schedule carries over from where it
        // left off (sigma_end), so continuing refines rather than re-broadening. Training time
        // is accumulated across rounds, excluding the time spent waiting at the prompt.
        double secs = 0.0, bmu_secs = 0.0, upd_secs = 0.0;
        auto run = [&](float sigma_start, uint32_t max_ep) {
            auto a = std::chrono::high_resolution_clock::now();
            TrainResult rr = train(dds, cb, sigma_start, max_ep);
            secs += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - a).count();
            bmu_secs += rr.bmu_seconds; upd_secs += rr.update_seconds;
            return rr;
        };

        TrainResult r = run(-1.0f, 0);                 // initial run: fresh schedule, cfg.max_epochs
        uint32_t total_epochs = (uint32_t)r.history.size();
        // Offer to continue only when the epoch cap was hit (not on convergence or a watchdog
        // restart flag). The σ schedule carries over from where it left off (sigma_end).
        while (r.stop_reason == StopReason::MaxEpochs && !r.history.empty()) {
            const EpochMetrics& m = r.history.back();
            std::printf("\n--- reached the epoch cap without converging (%u epochs) ---\n"
                        "  sigma=%.3f  QE=%.4f  KL=%.4f  stability=%.4f  TE=%.3f  dead=%.1f%%\n",
                        total_epochs, (double)r.sigma_end, m.qe, m.kl, m.stability,
                        m.te, 100.0 * m.dead_frac);
            uint32_t more = prompt_more_epochs();
            if (more == 0) break;
            r = run(r.sigma_end, more);                // continue from where σ left off
            total_epochs += (uint32_t)r.history.size();
        }

        // Watchdog flagged the run: topology stayed poor at small σ. Don't prompt/continue —
        // exit with a distinct code so a harness can restart with a slower σ rate.
        if (r.stop_reason == StopReason::RestartSlower) {
            std::fprintf(stderr, "needs restart with a slower sigma rate (topology poor)\n");
            return 3;
        }

        if (r.history.empty()) {
            std::printf("done: no epochs run\n");
        } else {
            const EpochMetrics& last = r.history.back();
            const char* tag = r.converged ? " (converged)" : " (stopped at cap)";
            std::printf("done: %u epochs in %.2fs (BMU %.2fs, update %.2fs) — final QE=%.4f  "
                        "stability=%.4f%s\n",
                        total_epochs, secs, bmu_secs, upd_secs, last.qe, last.stability, tag);
        }

        // Evaluate a held-out corpus: load, upload, run BMU, compute metrics, print JSON.
        // dead% is computed from the TRAINING set's last-epoch BMU (before resize) because
        // the val set may be too small to cover all neurons at large map sizes.
        if (!eval_corpus.empty()) {
            double dead = dead_fraction(cb);

            CsrDataset eds = CsrDataset::load(eval_corpus);
            if (eds.n_features() != cfg.n_features)
                throw std::runtime_error("--eval-corpus feature count " +
                    std::to_string(eds.n_features()) + " != training corpus " +
                    std::to_string(cfg.n_features));
            DeviceDataset edds;
            std::vector<float> evals_buf;
            const float* ehvals = nullptr;
            if (!use_float) { /* binary */ }
            else if (eds.has_values()) { ehvals = eds.values(); }
            else { evals_buf.assign(eds.n_nonzeros(), 1.0f); ehvals = evals_buf.data(); }
            edds.upload(eds.row_ptr(), eds.col_idx(), eds.n_samples(), eds.n_nonzeros(), ehvals);

            cb.resize_samples(eds.n_samples());
            launch_bmu(edds, cb);
            cudaDeviceSynchronize();

            double te   = topographic_error(cb);
            double cqe  = cosine_qe(cb, edds.d_row_ptr);
            double eqe  = mean_bmu_dist(cb);
            std::printf("metrics:{\"qe_cosine\":%.6f,\"qe_euclidean\":%.6f,\"topographic_error\":%.6f,"
                        "\"dead_fraction\":%.6f}\n", cqe, eqe, te, dead);

            cb.resize_samples(ds.n_samples());
        }

        // Optionally persist the trained codebook — with disk guard.
        // For large codebooks (>8 GiB), use chunked streaming that needs only ~8 GiB GPU
        // temp instead of a full FP32 copy. Small codebooks use the original one-shot path.
        if (!save_weights.empty()) {
            const size_t cb_bytes = (size_t)cfg.n_neurons() * cfg.n_features * sizeof(float);

            struct statvfs st;
            size_t avail_disk = 0;
            std::string dir = save_weights;
            auto slash = dir.rfind('/');
            if (slash != std::string::npos) dir = dir.substr(0, slash);
            else dir = ".";
            if (statvfs(dir.c_str(), &st) == 0)
                avail_disk = (size_t)st.f_bavail * (size_t)st.f_frsize;

            if (avail_disk > 0 && cb_bytes > avail_disk * 9 / 10) {
                std::fprintf(stderr, "SKIP --save-weights: codebook %.1f GiB > free disk %.1f GiB\n",
                             (double)cb_bytes / (1ull << 30), (double)avail_disk / (1ull << 30));
            } else if (cb_bytes > (size_t)8 * (1ull << 30)) {
                cb.save_codebook_chunked(save_weights);
            } else {
                std::vector<float> W;
                cb.download_W_fm(W);
                save_codebook(save_weights, cfg.map_rows, cfg.map_cols, cfg.n_features, W.data());
                std::printf("saved codebook -> %s  (%u neurons x %u features, %zu floats)\n",
                            save_weights.c_str(), cfg.n_neurons(), cfg.n_features, W.size());
            }
        }
        write_bmu();                       // dump the final per-sample clustering if --dump-bmu set
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
