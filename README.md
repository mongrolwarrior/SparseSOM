# SparseSOM

A GPU-only (CUDA, sm_89 / RTX 4090) C++17 **sparse-binary self-organising map**. Extracted and
refactored from `MedSOM-Naive` down to one clean **feature-major** backend, domain-neutral (no MEDLINE
metadata baked in). This is the current SOM implementation under evaluation by **SparseBinEval**.

> Detailed implementation status and the task history are in [`TASKS.md`](TASKS.md). This README is the
> orientation / how-to-run.

## What it is
- **Feature-major codebook** — weights stored as `d_W_fm[v*K + k]` (feature-major, not node-major), with
  per-neuron L2 + precision-weighted score norms. This layout is the project's central algorithmic choice
  (the "novel feature-major" line; origin evidence in `MedSOM/MedSOM-Naive/BENCHMARK_RESULTS.md`).
- **BMU** (`launch_bmu`) — tile-based sparse-dense scoring (SpmmTile); top-2 neurons per sample, precision-
  weighted (`bmu_beta`), writing `d_bmu1`/`d_bmu2`/`d_bmu_dist`.
- **Batch-SOM update** (`launch_update_factorized`) — atomic scatter → σ-independent separable **box-blur**
  → `W = Num/Den`, via a shared-memory tiled transpose.
- **Training** (`train`) — endpoint σ schedule (σ₀→σ_min, exp/linear) driven by the *windowed relative
  Kaski–Lagus improvement* (decoupled from epoch count); σ is a box-filter half-width. A **watchdog**
  accelerates decay on a plateau-while-σ-large and flags `RestartSlower` if the KL path (topology) term
  hasn't fallen to ≤ `wd_path_frac` of its peak once σ is small. Stops on a refinement-regime metric
  plateau (Kaski–Lagus default, QE selectable) or `max_epochs`.
- **Lattice:** square, **planar (edge-clamped)** neighbourhood. (Hex/toroidal are deferred future work —
  see `SparseBinEval/docs/convergent_validity_and_comparison.md` §4.)

## Build
```bash
cmake -B build && cmake --build build      # needs CUDA toolkit, sm_89
ctest --test-dir build                     # 15 tests: bmu, factorized update, training, dataset, cli, …
```
Binaries land in `build/`: `sparsesom` (CLI), `som_inspect` (host-only codebook inspector), plus the test
exes (`sbsom`/`sparsesom_*`).

## Run
```bash
# train a map and save the codebook + per-sample BMU assignment
sparsesom corpus.sbcsr --rows 64 --cols 64 --sigma-init 32 --sigma-min 1.0 \
          --save-weights map.somw --dump-bmu bmu.u32

# assign-only against a saved map (no training), report QE, dump BMUs
sparsesom corpus.sbcsr --load-weights map.somw --eval --dump-bmu bmu.u32

# inspect a saved map without a GPU
som_inspect map.somw
```
Key flags: `--rows/--cols`, `--epochs`, `--sigma-init/--sigma-min/--sigma-sched/--sigma-rate`,
`--beta` (`bmu_beta`), `--min-nnz`, `--seed`, `--save-weights`/`--load-weights`, `--eval`, `--dump-bmu`.
Geometry is read from the file on `--load-weights`. `--dump-bmu PATH` writes `uint32[n_samples]` in corpus
row order — **the clustering** consumed by the SparseBinEval scoring harness.

## Formats
- **`.sbcsr`** corpus — self-describing sparse-binary CSR (`uint32` row_ptr, `uint16` col_idx; binary
  presence, no values). `csr_convert` ingests a generic `.csr` and binarises it. (Format also documented in
  `SparseBinEval/data/README.md`.)
- **`.somw`** codebook — self-describing feature-major weights (magic `SSOMW1`/legacy `SBSOMW1`; the current
  binary loads both).

## Notes / known gaps
- No direct `medline_extract` reader — convert its 4-file layout to `.csr`/`.sbcsr` first (deliberate; see
  `TASKS.md` backlog).
- Metrics: QE, KL, stability, topographic error, dead-neuron fraction. No NMI (no per-sample labels in the
  core — labelling/scoring lives in SparseBinEval).
- Single-GPU only (sharding deliberately stripped; the distributed line is in `MedSOM`/`sombench`).
- The training watchdog **refuses to collapse to σ→0**; to produce a k-means/VQ control, cap epochs before
  the restart fires (see memory `som-scoring-gotchas`).

## Related
- **StandardSparseSOM** — a cuSPARSE node-major baseline for the comparison.
- **SparseBinEval** — the evaluation harness (map-size sweep, somoclu/StandardSparseSOM comparison,
  hierarchy-recovery scoring).
- **sombench/HANDOVER.md** — the scientific findings baseline.
