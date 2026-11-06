# Claims as targets

Each row: paper claim -> expected value -> tolerance -> command -> where to read the
result -> paper source. Expected values are this repository's numbers (reference tables
under `experiments/*/expected/`). Commands from the repo root, prefixed by `uv run`.
Status as in `reproduce/README.md`; only "done" and "running" rows can be checked now.

Tolerance shorthand: **exact** = identical or <= 1e-6; **1 std** = config mean within one
cross-fold (MUTAG) or cross-seed (ZINC) std of the reference runs; **MC** = Monte-Carlo noise.

## Features and hardware (App. D)

| Claim | Expected | Tolerance | Command | Read | Source |
|---|---|---|---|---|---|
| Released features equal the correlations recomputed from the raw counts | max abs diff 0 (emulation), 1.5e-8 (QPU) | exact | `python experiments/generation/validate_raw.py` | `experiments/generation/expected/validate_raw.json` | App. D.1 |
| Re-emulated correlations are statistically consistent with the stored ones | z-score mean ~0, std ~1.0 (AFM 1.01, FM 1.04) | std in [0.95, 1.08] | `--extra generate python experiments/generation/validate_emulation.py` | `expected/validate_emulation.json` | App. D.1 |
| ZINC\*_subset = 11 368 graphs | exactly the released ids | exact | `python experiments/generation/zinc_subset.py` | `expected/zinc_subset.json` | Sec. 2.2 |
| Temporal PCA, ZINC AFM: one-body / two-body PC1 | 80.6 %, 59.4 % (FM 80.9 %, 76.9 %) | <= 1.1e-5 | `python experiments/temporal_pca/run.py --dataset zinc` | `results/temporal_pca/pca_zinc.json` | Fig. 4, App. D.2 |
| Shot budget at N = 18 | 875 | exact | `python experiments/shot_budget/run.py` | `results/shot_budget/` | App. D.3 |
| QPU vs emulation per-graph Spearman, median (mean) | one-body 0.921 (0.910), two-body 0.870 (0.861); pooled 0.927 / 0.863 | exact | `python experiments/qpu_vs_emulation/run.py` | `results/qpu_vs_emulation/fig5.json` (reference `experiments/qpu_vs_emulation/expected/fig5.json`) | Fig. 5, App. D.4 |

## Intrinsic benchmark (Sec. 3.1, Tables 1, 11, 12) - done

References: `experiments/intrinsic/expected/results.csv` (MAE or accuracy, mean over 5
folds, population std) and the rendered tables next to it. Compare a run with
`python experiments/intrinsic/compare.py diff <results_dir> experiments/intrinsic/expected/results.csv`.
Tolerance: per cell within ~1 fold std, but GPU non-determinism plus early stopping puts
a minority of cells beyond 1 std; judge by the fraction of cells and by the table
rankings, not single cells.

| Claim | Expected | Source |
|---|---|---|
| AFM alone beats RRWP and RRWP+L on degree (MAE x10^2) | AFM 1.26, RRWP+L 6.98, RRWP 14.4 | Table 1 |
| Hybrid gains on degree: RRWP -> RRWP+AFM -> RRWP+FM+AFM | 14.4 -> 1.18 -> 1.25 (tie) | Table 1 |
| Local connectivity, same chain (x10^2) | 10.0 -> 6.07 -> 5.14 | Table 1 |
| Node cycle: RRWP+AFM < RRWP+FM | 0.36 vs 0.48 | Table 1 |
| Eccentricity: RRWP+FM < RRWP+AFM | 1.48 vs 1.77 | Table 1 |
| Strengthened baseline, degree MAE: RRWP+L+Ring / +Q / +Geo / +Geo+Q | 0.062 / 0.009 / 0.008 / 0.009 (tie) | ZINC strengthened-baseline table |
| Strengthened baseline, +Geo+Q wins on betweenness and bridge node | 0.018, 0.995 | idem |
| Residual table: both residuals win on bridge node and edge betweenness; betweenness and eccentricity tie | 0.996 / 0.003; 0.017, 0.631 | ZINC residual table |
| MUTAG with hardware features (Table 12) | HW-AFM bridge node 0.92, degree 7.8e-2 | Table 12 |

Command: `experiments/intrinsic/submit_grid.sh` (~73 A100 GPU-h), then
`python -m qgf.intrinsic.tables --results_dir results/intrinsic/grid --out_dir results/intrinsic/tables`.
Single cell: `python -m qgf.intrinsic.run --dataset MUTAG --pe_type RRWP_FM_AFM --benchmark degree_prediction --output_dir results/intrinsic/grid`.

## Reconstruction (Sec. 3.2, App. H) - done

| Claim | Expected | Tolerance | Command | Read | Source |
|---|---|---|---|---|---|
| Extended-basis ceiling, MUTAG RF R^2 | AFM 0.917 (8.3 % unexplained), FM 0.860 (14.0 %) | 0.25 fold std (RF), 1.5e-3 (OLS) | `python experiments/reconstruction/run.py ceiling` | `results/reconstruction/ceiling/ceiling_summary.csv` | Sec. 3.2 |
| Initial linear ceiling, OLS walk AFM K = 1/7/21/50 | MUTAG 0.597 / 0.782 / 0.825 / 0.832; ZINC 0.586 / 0.770 / 0.778 / 0.779 | <= 1e-8 | see `experiments/reconstruction/README.md` (sweep `--probes ols`) | `expected/table13_ols_time_averaged.csv` | Table 13 |
| Nonlinear probes, FM walk+Lap / RRWP+Lap, K = 21 | 0.847 / 0.840 (graph-level folds) | 1 fold std | sweep `--probes random_forest` | `expected/tables14_15_nonlinear_probes_mutag.csv` | Tables 14-15 |
| Short times are linear in adjacency powers | R^2 >= 0.9997 for t <= 30 ns | exact | `python -m qgf.reconstruction.run short-time --max-nodes 12 ...` | `expected/short_time_verdict.csv` | Sec. 3.2 |
| Residual spectrum (energy centroid, AFM / FM residual) | 1.284 / 0.379 | 1e-3 | `python experiments/reconstruction/run.py residual-analysis` | `expected/residual_spectral_centroid_zinc.csv` | residual analysis |
| Residual not explained by classical dynamics (dTWA), d >= 4 | R^2 ~0.000 (AFM), 0.004 (FM) | 0.01 | idem | `expected/residual_classical_test_zinc.csv` | residual analysis |
| Information retention of the residual (from stored per-task scores of the 15 structural tasks) | graph 97.4 %, node 82.5 %, edge 42.4 % | 0.5 pt | `python experiments/reconstruction/run.py retention --residual-results experiments/reconstruction/inputs/retention_residual.csv --full-results experiments/reconstruction/inputs/retention_full.csv` | `expected/retention_summary_zinc.csv` | residual analysis |

## Downstream GT (Sec. 4.1, Tables 2-3) - done

Tolerance **1 std**. Command `python experiments/downstream_gt/run.py --dataset <mutag|zinc> --configs paper`,
then `python experiments/downstream_gt/analyze.py`; read
`results/downstream_gt/<ds>_summary.csv` (compare with `experiments/downstream_gt/expected/<ds>_summary.csv`).

| Claim | Expected | Source |
|---|---|---|
| MUTAG GT: RRWP / RRWP+AFM+FM (column best) | 0.854 +- 0.060 / 0.874 +- 0.081 | Table 2 |
| MUTAG GT: AFM / FM alone | 0.830 / 0.851 | Table 2 |
| MUTAG GT: RRWP+AFM_stat+FM_stat (hybrid champion) | 0.862 | Table 2 |
| MUTAG GT: Laplacian / RRWP+Laplacian | 0.807 / 0.868 | Table 2 |
| MUTAG GT hardware: HW-AFM / HW-AFM_stat / RRWP+HW-AFM / RRWP+HW-AFM_stat | 0.784 / 0.778 / 0.854 / 0.849 | Table 2 |
| MUTAG GT geometry baseline: RRWP+L+Rings+r,J -> + AFM + FM | 0.8736 -> 0.8818 | geometry-baseline table |
| ZINC GT MAE: No Data / RRWP | 0.317 +- 0.018 / 0.113 +- 0.003 | Table 3 |
| ZINC GT MAE: AFM / FM / AFM+FM | 0.163 / 0.183 / 0.154 | Table 3 |
| ZINC GT MAE: RRWP+AFM_stat (column best) / RRWP+AFM_stat+FM_stat | 0.108 +- 0.007 / 0.111 +- 0.005 | Table 3 |
| ZINC GT MAE: Laplacian / RRWP+Laplacian | 0.232 / 0.116 | Table 3 |
| Other backbones (GCN, GAT, GINE, Transformer, GeneralConv) | see the next section | Tables 2-3 |

## Downstream GNNs (Tables 2-3, geometry table) - done (exact)

Command `python experiments/downstream_gnns/run.py --dataset <mutag|zinc>` (`submit.sh`),
then `python experiments/downstream_gnns/analyze.py --expected experiments/downstream_gnns/expected`.
Reference `expected/<ds>_{summary,trials}.csv` = the paper runs (mean, std ddof 1 over trials;
hardware rows of Table 2: `std_fold`). Tolerance: **bit-identical** trials with the pinned
environment `reproduce/paper_env` (`GNN_ENV=paper`) on AMD EPYC 7742 CPU nodes, 1 thread per
process for MUTAG, 4 for ZINC (1 for the Laplacian / geometry rows and 30 resubmitted v2
trials; list in the experiment README). Re-run: MUTAG 5500 / 5500, ZINC 450 / 450 identical,
so every cell, ordering and claim equals the paper. The project environment gives
statistically equivalent trials (MUTAG within 0.34 std per cell, ZINC mean 0.55 std) but
near-tied orderings can change there.

| Claim | Expected (paper runs) |
|---|---|
| MUTAG GCN / GAT / GINE / Transformer / General, RRWP | 0.849 / 0.862 / 0.861 / 0.869 / 0.868 |
| MUTAG Table 2 column best is an RRWP + quantum hybrid | 5/5 |
| MUTAG RRWP+AFM_stat+FM_stat >= RRWP | 5/5 |
| MUTAG geometry table: + AFM + FM improves RRWP+L+Rings+Geometry | 5/5 |
| MUTAG RRWP+HW-AFM_stat within 0.01 of RRWP+AFM_stat | 5/5 |
| ZINC GINE RRWP / RRWP+AFM_stat+FM_stat (MAE) | 0.117 / 0.116 |
| ZINC Table 3 column best is an RRWP + quantum hybrid | 5/5 (hybrid-vs-RRWP gaps <= 0.014 MAE, of the order of the 5-seed noise) |
| ZINC geometry baseline: + AFM + FM worsens RRWP+L+Rings+Geometry | 5/5 |
| Laplacian PE row (paper) | v2 run, 1-dim Laplacian (GCN 0.747); the port's 4-dim v3 reference: GCN 0.800 |

## Statistics (App. G) - done

Command: `--extra stats python -m qgf.stats.pe_benchmark --trials experiments/stats/inputs/gt_trials.csv experiments/stats/inputs/other_backbones_mutag_trials.csv --output-dir results/stats`;
read `results/stats/stats_report.md` (reference `experiments/stats/expected/stats_report.md`).
The exact MUTAG GNN re-run gives the same input trials (max difference 0).

| Claim | Expected | Tolerance |
|---|---|---|
| Hybrid champion ties or wins in 10/12 cells, one-sided sign test | p = 0.0193 | exact |
| Bayesian sign test, ROPE +-0.5 pooled SD (P_hyb, P_rope, P_cl) | 0.019, 0.981, 0.000 | MC +-0.005 |
| Signed-rank on 300 MUTAG pairs, ROPE +-0.01 F1 | 0.907, 0.000, 0.093 | MC +-0.005 |
| Hierarchical over 6 architectures, ROPE +-0.01 F1 | 0.240, 0.727, 0.033 | MC +-0.02 |
| Pooled mean F1 difference [95 %] | 0.0050 [-0.0028, 0.0128] | exact |

## Placement sensitivity - done

Command: `python experiments/layout_sensitivity/analyze.py data/raw/layout_sensitivity`
(stored study) or the Blade re-run (`--extra generate python experiments/layout_sensitivity/run.py ...`).

| Claim | Stored study | Blade re-run | Tolerance |
|---|---|---|---|
| Similarity, jitter s = 0.02 | 0.819 +- 0.076 | 0.813 | 0.01 on the mean |
| Similarity, s = 0.05 | 0.607 | 0.596 | 0.02 |
| Alternative placements | 0.558 | 0.585 | not expected to match exactly (different embedder) |
| Node-permutation floor | -0.001 | 0.001 | 0.01 |

## Synthetic ladder graphs (pinned vs frustrated) - done

Command: `python experiments/ladder/run.py` (`--full --perm 100` for every family). Data:
`ladder_{afm,fm}`, 394 RRWP-2-blind ladders, 98 blind groups, 18-26 atoms. Logistic
regression, 5 folds grouped by blind group x 3 repeats (stored folds); ROC-AUC mean ± std
over the 15 test folds. Reference: `experiments/ladder/expected/`. Tolerance: exact
(deterministic; the stored folds make it independent of the scikit-learn release).

| Claim | Expected | Source |
|---|---|---|
| Pooled RRWP-2 and ring features are blind by construction | ROC-AUC 0.500, perm p 0.485 / 0.614 | ladder table |
| Best classical combination (RRWP + Lap + Ring + Geometry) stays weak | 0.680 ± 0.065 (accuracy 0.641) | ladder table |
| AFM quench alone separates pinned from frustrated | 0.914 ± 0.026 (accuracy 0.847) | ladder table |
| AFM + FM, and classical + AFM + FM | 0.941 ± 0.022, 0.941 ± 0.023 | ladder table |
| Quantum gain over the classical combination | +0.26 ROC-AUC | ladder table |
| Ideal ground state (exact DP) | 1.000 | ladder table |

## ESA gated fusion (Table 4) - done (exact)

Command `ESA_ENV=paper NTASKS=16 PARALLEL=4 sbatch --array=0-15 experiments/esa/submit.sh
--configs paper` (pinned environment `reproduce/paper_env`: torch 2.4.0+cu121, xformers
0.0.27.post2, Lightning 2.5.3; `--attention xformers`), then
`python experiments/esa/analyze.py --results results/esa_exact --expected experiments/esa/expected`.
Reference `expected/mutag_trials.csv` / `mutag_summary.csv` (= paper, 15 cells and 7 gates).
Metric: binary F1 of the positive class (the caption says macro), std ddof 0. Tolerance:
**bit-identical** trials in the pinned environment; the project environment (SDPA attention)
gives close but not identical numbers.

| Claim | Expected (paper runs) | Re-run |
|---|---|---|
| All 750 trials (15 cells x 10 folds x 5 seeds) | stored values | 750/750 bit-identical |
| RWSE\* baseline | 0.905 ± 0.045 | identical |
| Gated RWSE\*+AFM+FM best overall, gated +HW-AFM second | 0.922 / 0.920 | identical |
| Gated fusion >= concatenation, every quantum input except FM | 6/7 | identical |
| Learned gates (init 0.88) | 0.816-0.887 | identical |

HW rows: `mutag_hw_afm` at 305 / 512 / 4000 ns (config `hw_afm_3t*`), read in the ZZ
convention and clipped to [-1, 1] as the original loader did (the clip is active).
Gated-vs-concat differences are within about one std per cell.

## Ablations (Tables 8-10, Fig. 6) - done (exact)

Command `python experiments/ablations/run.py` (`submit.sh`; exact: `ABL_ENV=paper`, pinned
environment, 4 threads per process, AMD EPYC 7742), then
`python experiments/ablations/analyze.py --results results/ablations --out results/ablations/tables`.
Data `zinc_tiny_{afm,fm}` (earlier 6 us protocol). Cells pool training settings and repeated
seeds as the original (`expected/trials.csv`). Exact re-run: 1360 / 1360 distinct trials
bit-identical; Table 10 identical; Tables 8-9 differ in 5 rows (<= 0.013 MAE) only through 30
pooled rows of the original sweep `20260401_140818_GCN_all_sizes`, whose training code changed
while it ran (its clean redo `GCN_rerun_fixed` is reproduced exactly); left as published.

| Claim | Expected (paper runs) |
|---|---|
| Table 8 baseline GCN / GAT | 0.520 / 0.480 |
| Table 8, AFM<sub>1b</sub> best for GCN, AFM<sub>2b</sub> best for GAT | 0.443 / 0.449 |
| Table 8, quantum rows below the baseline | 5/6 per model (exceptions GCN AFM<sub>2b</sub>, GAT FM) |
| "Large is the best size" (Tables 9-10) | not literal: large best in 1/7 (GCN), 2/7 (GAT) rows; row-average minimum at huge; medium to huge equivalent, larger worse |
| Fig. 6, RRWP+AFM gap closes with width (MAE D = 64 -> 256) | 0.121 -> 0.111 (test-set selection; not re-run) |

Only the GCN AFM<sub>1b</sub> / AFM gains exceed one pooled std; the original code has none of
the three protocol defects (checked).

## Fig. 2, toy recovery - done

`python experiments/reconstruction/fig2.py --out results/reconstruction/fig2`: all 720 values
equal the published run (1e-8). AFM recovery stays >= 0.8 at 400 ns for K >= 3; FM drops to
~0 after ~70 ns for K = 1. That FM panel used `delta = delta_inf`; with the paper's FM rule
(`--fm-delta paper`) FM stays at R^2 0.5-0.85: the caption's "not recovered anymore" does
not hold for FM at `2 delta_inf`.

## Exact non-unit-disk Hamiltonian (MUTAG, 173 graphs) - done (exact)

Generated features == original to 3e-8 (float32). Exact re-run (pinned environment, 1 thread,
AMD EPYC 7742): 1400 / 1400 trials bit-identical. On the 173 MUTAG graphs with N <= 24, neither
exact (noise-free) AFM + FM features nor the exact graph Hamiltonian `J_NN A` improve on the
strong classical baseline across 7 GNN backbones (paired Wilcoxon p >= 0.086). Not a paper claim.

## Not checkable yet

Physics (Figs. 1, 3): to be decided. Correlation of residuals with graph invariants: internal
source recovered (pooled partial Spearman, ZINC; the 0.46 / 0.33 / 0.17 means were computed by
hand from the per-task peaks), not ported.
