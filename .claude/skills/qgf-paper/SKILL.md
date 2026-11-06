---
name: qgf-paper
description: Use for anything about the paper "Neutral-Atom Quantum Features as Complementary Structural Encodings for Graph Learning" (NeurIPS 2026) and this repository - explaining the method, physics intuition, results or limitations; locating the code, data or command behind a table, figure or number; reproducing a paper table/figure and checking it against the expected values.
---

# qgf-paper

Three modes. Pick from the request; ask only if it is genuinely ambiguous.

| Request looks like | Mode |
|---|---|
| "what is / why / how does ... work", "what does the paper show" | explain |
| "where is the code / data for X", "which command gives Table N" | guide |
| "reproduce / check / re-run Table N, Fig. M, claim X" | reproduce |

Sources of truth, in this order:
1. `references/claims.md` - every quantitative claim with expected value, tolerance, command.
2. `reproduce/README.md` - paper item -> command -> inputs -> output -> status;
   `reproduce/walkthrough.ipynb` recomputes each item from the precomputed data.
3. `experiments/<item>/README.md` - details and protocol notes.
4. `references/paper-summary.md`, `references/glossary.md` - prose and terms.
5. `AGENTS.md` - data conventions (ZZ = 4C, time-major, QPU estimator).

Never invent a number. If a value is not in these files, say so.

## Mode: explain

1. Read `references/paper-summary.md` (and `glossary.md` for terms).
2. Answer at the level asked: one-paragraph intuition first, then details on request.
3. Keep claims calibrated as the paper does: downstream gains are small and mostly within
   cross-fold noise but sign-consistent; the reconstruction result is an empirical
   ceiling for the tested descriptor families, not a proof of classical hardness; no
   quantum-advantage claim is made.
4. When a number is quoted, give its source item (e.g. "Table 1") and, if the repo
   reproduces it, the reproduced value from `claims.md`.
5. The paper (Garrigues, Sokolov, Hecker, Moro, Acheche; NeurIPS 2026) is cited from the
   README; summarise in your own words, do not paste long passages.

## Mode: guide

1. Find the item in `reproduce/README.md` (search by table/figure number or keyword).
2. Report: status, command, input datasets, output path, expected value, and the
   experiment README for details.
3. For code locations, point to the package under `src/qgf/` named in the experiment
   README (e.g. intrinsic probes: `src/qgf/intrinsic/deepset_probe.py`).
4. For data questions, use `AGENTS.md` (conventions) and `qgf.data` (`load`,
   `load_splits`, `verify`). Dataset names: `mutag_{afm,fm,hw_afm}`,
   `mutag_{afm,fm}_residual`, `ladder_{afm,fm}`, `zinc_{afm,fm}[_residual]`,
   `zinc1500_{afm,fm}`, `zinc_tiny_{afm,fm}` (ablations; splits),
   `mutag_exact[_nn]_{afm,fm}` (non-unit-disk, 173 graphs).
5. Status "coming" or "to be decided": say the item is not in the repository yet; do not
   improvise an implementation unless asked.

## Mode: reproduce (claims as targets)

Treat each paper claim as a test target with a tolerance.

1. **Pick the item.** Find it in `references/claims.md`; note the expected value, the
   tolerance and the command.
   If its status is not "done" or "running", stop and report that.
2. **Check the data.**
   ```bash
   uv run python -c "from qgf.data import verify; print(verify() or 'ok')"
   ```
   Missing files: `uv run python -m qgf.data.download <record>`; raw counts need
   `--files qgf_raw.tar` then `tar xf data/processed/qgf_raw.tar -C data`; published
   per-trial results: `--files qgf_results.tar qgf_results_additions.tar` (the additions
   hold the exact GNN / ESA / ablation / non-unit-disk trials).
   Extras: `uv sync --extra stats` (App. G), `uv sync --extra generate` (pulser, emulators,
   Blade).
3. **Estimate cost before running.** Cheap (< 15 min CPU): temporal PCA, shot budget,
   QPU vs emulation, stats, ceiling, residual analysis, intrinsic table generation from
   finished results. Expensive: intrinsic grids, downstream grids (MUTAG ~3 h on 8
   A100, ZINC ~1 day on 16 A100), ESA, ablations, non-unit-disk, placement re-run, any
   emulation. For expensive items
   first run the documented single-trial / single-cell command, and ask the user before
   launching a full grid. Use the Slurm wrappers next to `run.py` when on a cluster.
4. **Run** the exact command from `claims.md`, writing to `results/<item>/` (gitignored).
   Never overwrite `experiments/*/expected/`.
5. **Compare.** Parse the output file named in `claims.md`; compute
   `|reproduced - expected|` and compare with the tolerance (for trained models usually
   one cross-fold or cross-seed std; deterministic pieces exact or <= 1e-6; Monte-Carlo
   statistics +-0.005 / +-0.02). Comparison helpers exist:
   `experiments/intrinsic/compare.py diff <results> experiments/intrinsic/expected/results.csv`,
   `experiments/downstream_gt/analyze.py` (per-config mean / std, compare with
   `experiments/downstream_gt/expected/mutag_summary.csv`).
6. **Report** (short, markdown, to the user; save under `results/<item>/report.md` if
   asked):
   ```
   Item: <paper item>    Command: <command>
   Data check: ok | <problems>
   | quantity | expected | reproduced | |delta| | tolerance | pass |
   Verdict: reproduced / partially / not reproduced
   Notes: environment (GPU/CPU, seeds), anything unexpected
   ```

## Pitfalls

- `onebody_features` is `[T, N]`, `twobody_features` `[T, N, N]`: transpose to node-major
  before PyG batching; never decide by comparing shapes (N can equal T = 20).
- Correlations are `4 C`. Divide by `meta["zz_scale_applied"]` to get the original
  inputs when a pipeline is not scale-invariant.
- QPU data (`mutag_hw_afm`): 4 times only; one-body diagonal with the single-atom readout
  correction.
- Reconstruction reference instant = time index 7 (1536 ns).
- Trained models: ESA, downstream GNNs, ablations and non-unit-disk are bit-identical
  only in `reproduce/paper_env` on the original CPU / GPU and thread count (`AGENTS.md`);
  then compare trial by trial (tolerance 0). Elsewhere (and for the GT) trials differ run
  to run: compare means over folds/seeds, not single trials.
