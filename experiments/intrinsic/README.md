# Intrinsic information benchmark

Probes how much structural information each feature set carries, without
message passing: a DeepSet (per-entity MLP for node/edge tasks, sum-pooled
DeepSet for graph tasks) is trained on the features alone to predict 15
structural targets (5 node, 5 edge, 5 graph), 5-fold cross-validation over
graphs, on MUTAG (188 graphs) and ZINC\*_subset (11 368 graphs).

Code: `src/qgf/intrinsic/` (`metrics.py` targets, `pe_construction.py`
encodings + normalisation + feature builders, `deepset_probe.py` probes,
`data_loading.py` datasets, `run.py` CLI, `tables.py` tables).

## Paper items

| Paper item | Output (under `--out_dir`) | Expected (committed) |
|---|---|---|
| Table 1 (main, ZINC\*_subset, means) | `ZINC_small_table_main.{md,tex}` | `expected/ZINC_small_table_main.md` |
| Table 11 (appendix, ZINC\*_subset, mean ± std) | `ZINC_small_table_full.{md,tex}` | `expected/ZINC_small_table_full.md` |
| Table 12 (appendix, MUTAG, mean ± std) | `MUTAG_table_full.{md,tex}` | `expected/MUTAG_table_full.md` |
| Strengthened-baseline comparison (ZINC, mean ± SE) | `ZINC_quantum_comparison.{md,tex}` | `expected/ZINC_quantum_comparison.md` |
| Residual comparison (ZINC, mean ± SE) | `ZINC_residual_comparison.md` | `expected/ZINC_residual_comparison.md` |

`expected/results.csv` holds all (dataset, feature set, task) cells:
`raw_value` (accuracy or MAE, mean over folds) and `std_dev` (population std
over folds).

## Commands

```bash
# 1. caches (classical encodings + targets, ~5 min on 16 CPUs; ZINC local
#    edge connectivity and minimum cycle bases dominate)
uv run python -m qgf.intrinsic.run --dataset MUTAG      --output_dir results/intrinsic/x --prepare_only --workers 8
uv run python -m qgf.intrinsic.run --dataset ZINC_small --output_dir results/intrinsic/x --prepare_only --workers 16

# 2. full grid (Slurm job array, one A100 per element; re-run to fill gaps)
experiments/intrinsic/submit_grid.sh              # -> results/intrinsic/grid/
#    single cell, no Slurm:
uv run python -m qgf.intrinsic.run --dataset MUTAG --pe_type RRWP_FM_AFM \
    --benchmark degree_prediction --output_dir results/intrinsic/grid

# 3. tables (markdown always, --latex for the paper .tex)
uv run python -m qgf.intrinsic.tables --results_dir results/intrinsic/grid --out_dir results/intrinsic/tables --latex

# 4. compare with the expected numbers (|delta| / std of the reference, per cell)
uv run python experiments/intrinsic/compare.py diff results/intrinsic/grid experiments/intrinsic/expected/results.csv
```

## Protocol

- Quantum features are stored time-major and transposed to node-major
  (`(N, T)`, `(N, N, T)`) for every graph.
- 5-fold CV over graphs (seed 42); 10 % of the training graphs of each fold
  are held out for early stopping and checkpoint selection; the test fold is
  only scored.
- Min-max scaling to [-1, 1] (per column / timestep) is fitted on the training
  graphs of each fold. Random features are never normalised.
- HW feature sets use `mutag_hw_afm` (QPU, 4 times; the one-body diagonal
  uses the single-atom readout correction).

## Results (headline)

On ZINC\*_subset (Table 1), RRWP+FM+AFM is best or tied on most
non-saturated tasks (e.g. betweenness MAE 3.31e-2, edge betweenness 5.36e-3,
local connectivity 5.14e-2, diameter 0.32); AFM alone reaches degree MAE
1.26e-2. In the strengthened-baseline comparison, RRWP+L+Ring+Geo+Q keeps
clear wins on betweenness and bridge node. In the residual comparison, both
residuals keep clear wins on bridge node and edge betweenness. Full numbers:
`expected/*.md`.

## Validation

- Deterministic pieces (features after normalisation, all 15 targets) were
  checked against an independent implementation: bitwise for quantum, RRWP,
  Ring, summary statistics and targets; <= 1.1e-6 for Laplacian; <= 2e-6 for
  Interaction (atom positions are stored in float32).
- GPU training is not bitwise deterministic and early stopping amplifies it:
  compare re-runs per cell in units of the fold std (`compare.py diff`).

## Compute

Caches: ~5 min on 16 CPUs (ZINC). Grid on A100 (1 GPU per array element, 4 CPUs,
16 GB): 0.8 GPU-h (MUTAG) + 72 GPU-h (ZINC). ZINC edge tasks (2.9 M node
pairs) dominate; ~4 h wall with 32 concurrent GPUs.

## Notes

- `tables.py` keys results on the directory name (not on the json `pe_type`).
- The bar-chart figure of the internal script is not ported (not in the paper).
