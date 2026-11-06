# Downstream Graph Transformer benchmark (paper Tables 2-3, GT column; geometry table)

Code: `src/qgf/downstream/gt/` (port of the original internal implementation of the GT
model, its encodings and its MUTAG / ZINC benchmark runners).

- MUTAG: 10-fold CV stratified by (label, graph-size quartile), `random_state=42`; fold
  k = test, fold k+1 = validation (early stopping on val macro-F1), seeds 1-5.
  Reported: per fold, average of the 5 seeds; mean and std over the 10 folds
  (std ddof=0 in Table 2, ddof=1 in the geometry-baseline table).
- ZINC\*<sub>subset</sub>: fixed split 9475/946/947, seeds 0-9, early stopping on val MAE;
  mean and std (ddof=0) over seeds.
- Hyperparameters: Table 6 (`benchmark.HPARAMS`). All feature blocks are z-scored per
  dimension with training-split statistics; RRWP k = 20; Laplacian = 4 eigenvectors
  (+ outer products as pair features).

## Commands

```bash
# one trial (GPU strongly recommended; MUTAG ~30 s, ZINC ~2 h on an A100)
uv run python experiments/downstream_gt/run.py --dataset mutag --configs rrwp --folds 0 --seeds 1
# full grids (skips finished trials); results/downstream_gt/<dataset>/<config>/...
uv run python experiments/downstream_gt/run.py --dataset mutag --configs paper
uv run python experiments/downstream_gt/run.py --dataset zinc --configs paper
# optional Slurm wrapper: <n_jobs> <gpus_per_job> <procs_per_gpu> <run.py args>
experiments/downstream_gt/submit.sh 1 8 3 --dataset mutag --configs paper
experiments/downstream_gt/submit.sh 2 8 2 --dataset zinc --configs paper
# tables: results/downstream_gt/<dataset>_summary.csv
uv run python experiments/downstream_gt/analyze.py --results results/downstream_gt
```

Compute used for the reference reproduction: MUTAG grid (28 configs x 50 trials) ≈ 3 h
on one 8-GPU A100 node; ZINC grid (22 configs x 10 seeds) ≈ 1 day on two 8-GPU nodes.

## Paper item -> config -> expected numbers

`expected/mutag_summary.csv` (28 MUTAG configs) and `expected/zinc_summary.csv` (22 ZINC
configs) are the output of `analyze.py` on the reference run
(macro-F1 on MUTAG, MAE on ZINC; mean ± std, ddof 0).

| paper item | config | value |
|---|---|---|
| Table 2 GT, No Data | `no_pe` | 0.690±0.104 |
| Table 2 GT, RRWP | `rrwp` | 0.854±0.060 |
| Table 2 GT, AFM / FM | `af` / `fm` | 0.830±0.054 / 0.851±0.059 |
| Table 2 GT, RRWP+AFM+FM (best) | `rrwp_af_fm` | 0.874±0.081 |
| Table 2 GT, RRWP+AFMstat+FMstat | `rrwp_af_stat_fm_stat` | 0.862±0.053 |
| Table 2 GT, HW-AFM / HW-AFMstat | `hw_afm` / `hw_afm_stat` | 0.784±0.098 / 0.778±0.086 |
| Table 2 GT, RRWP+HW-AFM / RRWP+HW-AFMstat | `rrwp_hw_afm` / `rrwp_hw_afm_stat` | 0.854±0.074 / 0.849±0.072 |
| Table 2 GT, Laplacian PE | `laplacian` | 0.807±0.065 |
| Table 2 GT, RRWP+Laplacian | `rrwp_laplacian` | 0.868±0.071 |
| Geometry table, RRWP+L+Rings+Geometry | `rrwp_lap_ring_geo` | 0.8736±0.0518 (ddof 1) |
| Geometry table, + AFM + FM | `rrwp_lap_ring_geo_af_fm` | 0.8818±0.0549 (ddof 1) |
| Table 3 GT, No Data | `no_pe` | 0.317±0.018 |
| Table 3 GT, RRWP | `rrwp` | 0.113±0.003 |
| Table 3 GT, AFM / FM / AFM+FM | `af` / `fm` / `af_fm` | 0.163±0.004 / 0.183±0.010 / 0.154±0.006 |
| Table 3 GT, RRWP+AFM / RRWP+AFM<sub>stat</sub> (best) | `rrwp_af` / `rrwp_af_stat` | 0.111±0.004 / 0.108±0.007 |
| Table 3 GT, RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub> | `rrwp_af_stat_fm_stat` | 0.111±0.005 |
| Table 3 GT, Laplacian PE / RRWP+Laplacian | `laplacian` / `rrwp_laplacian` | 0.232±0.020 / 0.116±0.007 |

ZINC: every config is within 1 seed std of the stored per-seed results of the paper run
(`experiments/stats/inputs/gt_trials.csv`; max |delta|/std = 0.73, `rrwp_af_stat_fm_stat`
0.1107 vs 0.1067).

All other rows of Tables 2-3 (GT column) and the Random / time-mean variants are in the
expected CSVs.

## Checks

- Encodings: `tests/downstream/test_reference_features.py` checks the emulation-based
  encodings bit for bit (max abs diff 0.0) against a reference dump of the original
  implementation (24 MUTAG and 5 ZINC configs, including all graphs with N = T = 20; needs
  the internal dump made by `scripts/internal/downstream_reference_features.py`, skipped
  otherwise).
- Training is not bit-reproducible on GPU (AMP, atomic adds); re-runs agree within the
  cross-fold (cross-seed) noise.

## Notes

- **HW-AFM (QPU) features**: the `*hw_afm*` configs read `mutag_hw_afm`
  (readout-corrected jackknife; single-atom readout correction on the one-body diagonal).
- Early stopping on MUTAG monitors the validation macro-F1.
- Released correlations are in the ZZ convention (4 C). The original consumed ZINC and
  HW-AFM without the factor 4; the loader divides by `meta["zz_scale_applied"]` so the
  inputs are exactly those of the original (all blocks are z-scored, so this only
  matters through the 1e-6 std epsilon). The original MUTAG `clamp(-1, 1)` is then a
  no-op and is not reproduced (with ZZ-scaled HW-AFM it would have clipped values up to
  1.3).
- No train/val/test leakage: all normalisation statistics are fitted on the training
  split (training folds for MUTAG). The residual one-body features are transposed
  unconditionally (no N == T shape-check hazard); the released residual files are
  already time-major.
- Dropped (not in the paper): ground-state/HES/local-detuning pulses, XY, PTC_FM,
  `hw_afm_full_stat`, `maxabs` / `zscore_diag` normalisations, TensorBoard logging.
