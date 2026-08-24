# SparseSOM — Project Tasks & Status

SparseSOM is a GPU-only (CUDA, RTX 4090 / sm_89) C++17 sparse-binary
self-organising map, built by extracting and refactoring algorithms and data
structures from **MedSOM-Naive** (`/workspaces/MedSOM/MedSOM-Naive`) — stripping
the comparison/research scaffolding down to one clean feature-major backend, and
keeping the original MEDLINE configuration only as defaults.

## Current capabilities (as of 2026-06-18)

> **Update:** BMU computation (Task 3) and the training loop (Task 4) are now implemented.
> The full SOM trains end-to-end on the GPU: randomise → (BMU → factorized update → σ-decay
> → convergence) over epochs.

Two static libraries (`sparsesom_core`, `sparsesom_featuremajor`) plus a CLI skeleton,
under a working CMake/CTest harness.

- **Feature-major codebook (`FeatureMajorCodebook`)** — GPU codebook stored only
  in feature-major layout `d_W_fm[v*K + k]`, with per-neuron norm arrays.
  `randomise()` (reproducible init scaled to `(0, 1/√V)`) and `recompute_norms()`
  (L2 norm + precision-weighted score norm `Σ wᵛ^(β+2)`).
- **BMU computation (`launch_bmu`)** — for every sample, finds the best and second-best
  neurons by precision-weighted score against the feature-major codebook (top-2 reduction),
  writing `d_bmu1`, `d_bmu2`, and the Euclidean `d_bmu_dist`. Tile-based sparse-dense scoring
  (SpmmTile) so a warp reads `W_fm[v*K + k]` coalesced and reuses it across the tile's samples
  sharing feature `v`.
- **Training loop (`train`)** — drives the full batch-SOM over epochs: each epoch assigns
  every sample (BMU), measures QE/stability/KL/TE/dead, and updates the codebook at the current
  σ. **σ is an endpoint schedule (σ₀→σ_min), decoupled from epoch count** — it interpolates
  (exponential default / linear) over a progress variable driven by the *windowed relative
  Kaski–Lagus improvement* (σ stays wide while KL drops fast → global ordering; contracts as
  improvement flattens → refinement). σ is a box-filter half-width. A **watchdog** accelerates
  the decay on a plateau-while-σ-large, and flags `RestartSlower` if the KL path (topology) term
  hasn't fallen to ≤ `wd_path_frac` of its peak once σ is small. Stops (Converged) on a metric
  plateau in the refinement regime — Kaski–Lagus by default, QE selectable — or `max_epochs`.
  `train()` returns `TrainResult{history, stop_reason, sigma_end}`; resumable via `sigma_start`.
- **Factorized batch-SOM update (`launch_update_factorized`)** — accumulate
  (atomic scatter to BMU neuron) → σ-independent separable box-blur → `W = Num/Den`
  written into the feature-major codebook via a shared-memory tiled transpose
  (coalesced read and write); numerator streamed in feature chunks.
- **Host dataset loader (`CsrDataset`)** — builds a sparse-binary CSR corpus from in-memory
  arrays or a self-describing `.sbcsr` binary file (load/save), with `filter_min_nnz` to drop
  sparse samples. Generic (no MEDLINE pmid/year/vocab metadata); feeds `DeviceDataset::upload`
  and `SomConfig::n_features`.
- **CSR converter (`csr_convert` + `CsrDataset::from_csr_file`)** — ingests a generic
  standard CSR binary (`.csr`: int32 offsets/indices + optional float32 values, cuSPARSE-like),
  binarises it (positive entries → present features), narrows types, drops rows with
  `< min_nnz` positive features (default 5), and writes `.sbcsr`.
- **Sparse-binary device dataset (`DeviceDataset`)** — uploads a CSR presence
  matrix (`uint32` row_ptr, `uint16` col_idx); features are binary, no values/weights.
- **Generic config (`SomConfig`)** — map geometry, feature dimension (`n_features`),
  `box_passes`, `bmu_beta`; dataset-agnostic, defaulting to the MEDLINE headline
  config (350×350 map, 30000 features) via a named `defaults` block.
- **CLI (`sparsesom`)** — `sparsesom <corpus.sbcsr> [--rows --cols --epochs --sigma-init --sigma-min
  --decay --beta --min-nnz --seed --save-weights --load-weights --eval]`: loads a corpus, builds
  or restores the codebook, trains, reports progress + a final summary, and can persist the map.
  If a run hits the epoch cap without converging it prints a map summary and (interactively)
  prompts to continue for more epochs — σ carries over; a non-TTY never blocks. Full pipeline:
  `CsrDataset::load → DeviceDataset::upload → FeatureMajorCodebook + train`.
- **Codebook persistence, reload + inspector** — `--save-weights` writes the trained
  feature-major codebook to a self-describing `.somw` file; `--load-weights` restores one onto
  the GPU (`upload_W_fm`, geometry taken from the file) to refine or, with `--eval`, assign a
  corpus and report QE without training (`evaluate_qe`). `som_inspect <weights.somw>` reports
  map geometry, weight range, per-neuron L2 norms, and the dead-neuron count — no GPU needed.
- **Build/test** — `ctest` covers the saxpy smoke test, factorized-update correctness
  (single/multi-chunk), BMU vs a host reference, the training loop, the dataset loader, an
  end-to-end `pipeline` (file → train) integration test, and a `cli` test driving the binary.

## Known gaps

- No direct `medline_extract` reader — its 4-file layout needs an external step to a generic
  `.csr` (or `.sbcsr`) first; a small adapter could read it directly.
- Metrics reported: QE, KL, stability, topographic error, dead-neuron fraction. No NMI (would
  need per-sample labels, which the pipeline has no notion of yet).
- No dead-neuron seeding/jitter, so a too-wide σ on a small map can collapse irrecoverably
  (sized schedules avoid this; revisit if needed).
- Single-GPU only (sharding/distributed deliberately stripped).

## Task list

### Done
- [x] **Task 1 — Feature-major codebook + factorized update** (PR #1).
      `FeatureMajorCodebook`, `launch_update_factorized`, `DeviceDataset`, trimmed `SomConfig`.
- [x] **Task 2 — Genericise the SOM** (PR #2). MEDLINE values become named
      defaults; geometry is fully config-driven.

- [x] **Task 3 — BMU computation** (`bmu.cu`, `launch_bmu`). Extracted MedSOM's
      SpmmTile BMU as tile-based sparse-dense scoring against the feature-major
      codebook, producing `d_bmu1` + `d_bmu_dist`. Refactored to top-1 only (dropped
      `bmu2`, which only fed the not-yet-extracted topographic-error metric), no
      `d_perm`/mini-batch/sharding, general-β scoring. Verified against a host
      reference for β=0 and β=1 (`tests/test_bmu.cpp`).

- [x] **Task 4 — Training loop** (`training.{hpp,cu}`, `train`). Epoch driver wiring
      `launch_bmu` → `launch_update_factorized` with geometric σ-decay and a QE-flat +
      stability convergence check (gated on σ reaching `sigma_min`). No lr (batch-mean
      update), no mini-batches/jitter. Verified end-to-end on a separable corpus
      (`tests/test_training.cpp`): QE → 0, converges.

- [x] **Task 5 — Dataset loader** (`dataset.{hpp,cpp}`, host `CsrDataset`). Generic
      sparse-binary CSR: in-memory construction, `.sbcsr` load/save (self-describing
      header), `filter_min_nnz`. No MEDLINE metadata. Tested: round-trip, filter, error
      handling (`tests/test_dataset.cpp`).

- [x] **Task 6 — End-to-end CLI** (`cli/main.cpp`). `sparsesom <corpus.sbcsr> [opts]`: load →
      size codebook → train → report. Covered by a `pipeline` integration test (file →
      train) and a `cli` test that runs the binary (`tests/test_pipeline.cpp`).

- [x] **Task 7 — CSR converter** (`csr_convert`, `CsrDataset::from_csr_file`). Generic
      standard CSR (`.csr`) → `.sbcsr`: binarise (keep positive entries), narrow types,
      `filter_min_nnz` (default 5). Tested: positive-element selection, filtering, type
      narrowing, errors (`tests/test_csr_convert.cpp`); demonstrated `.csr` → train.

- [x] **Task 8 — Codebook persistence + inspector** (`codebook_io.{hpp,cpp}`, `--save-weights`,
      `som_inspect`). `.somw` save/load (self-describing header), GPU download via
      `download_W_fm`, host-only inspector (geometry, weight + per-neuron-norm stats, dead
      count). Tested: format round-trip/errors + a `cli`→`inspect` CTest chain.

- [x] **Task 9 — `.somw` GPU re-upload** (`upload_W_fm`, `evaluate_qe`, `--load-weights`,
      `--eval`). Restore a saved codebook onto the GPU (geometry from the file, norms rebuilt)
      to refine or assign-only. Verified: a restored map reproduces identical BMU assignments
      (`tests/test_reupload.cpp`), plus an `eval` CTest driving the binary.

- [x] **Task 10 — Selectable stop criterion (Kaski–Lagus default / QE alternative).**
      Step 1: `StopCriterion` enum + windowed QE-plateau (relative change < ε for
      `plateau_window` epochs, gated on σ ≤ σ_min). Step 2: re-added `bmu2` via a top-2 BMU
      reduction + `d_bmu2`. Step 3: Kaski–Lagus relational measure (`kaski_lagus.cu`) — per
      sample, QE + a capped 4-connected grid path bmu1→bmu2 in plain-Euclidean input distance
      (commensurable with QE; β-weighted variant surfaced in config, not built), over a
      monitoring subset; now the **default** trigger, QE selectable. Step 4: topographic error
      + dead-neuron fraction added to `EpochMetrics` and the per-epoch log as **non-triggering
      diagnostics**. Each step verified against hand-computed values.

- [x] **Task 11 — Interactive continue-training prompt.** `train()` now returns `TrainResult`
      {history, converged, sigma_end} and takes `sigma_start`/`max_epochs` params so a run can
      **continue where σ left off** (no re-broadening). When the CLI hits the epoch cap without
      converging, it prints a map summary (QE, KL, stability, TE, dead%) and prompts for more
      epochs (default **20**, a number, or stop). Non-interactive-safe: a non-TTY stdin always
      stops (never blocks tests/cron). Resume primitives covered by `test_training`.

- [x] **Task 12 — De-MEDLINE-ify terminology.** Renamed `n_vocab`→`n_features` and
      "articles"→"samples" (identifiers + comments) across the whole codebase; the public API
      and docs are now domain-neutral. On-disk `.sbcsr`/`.somw` layouts are unchanged (header
      field *names* only, not bytes). Build + 15/15 tests green.

- [x] **Task 13 — KL-driven endpoint σ schedule + watchdog.** Replaced per-epoch geometric
      σ-decay with an endpoint interpolation (σ₀→σ_min, exp default/linear) driven by the
      *windowed* relative Kaski–Lagus improvement (decoupled from epoch count; σ in box-half-width
      units). Watchdog: accelerate on plateau-while-σ-large; flag `RestartSlower` (trajectory-
      relative: KL path term must fall to ≤ `wd_path_frac` of its peak at small σ). `TrainResult`
      gained `stop_reason`; CLI exits 3 on restart and only continue-prompts on `MaxEpochs`. New
      config: `sigma_schedule/sigma_rate/sigma_g_ref/sigma_window/sigma_accel/wd_path_frac`
      (removed `sigma_decay_rate`). Schedule helper unit-tested. (Harness wiring = SparseBinEval,
      next.)

### In progress
- _(none — pick the next backlog item)_

### Backlog
- [ ] Real-data ingestion (deferred until needed). A `medline_extract` (4-file) corpus reaches
      SparseSOM via `.sbcsr`. **Decision:** do NOT bake a MEDLINE reader into the generic core
      — prefer a standalone converter script (the format is already uint16-col / uint32-offset
      binary CSR) or have `medline_extract` emit `.sbcsr`/`.csr` directly. The in-core
      `from_medline_extract` adapter is only worth it if MEDLINE becomes a first-class input.
