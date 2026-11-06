# Classical reconstruction ceiling and residual analysis

Pooled regression of the quantum correlations C_ij(t) (one shared coefficient vector over
all pairs of all graphs) on classical pair descriptors. The R^2 is an expressivity
ceiling, not a learning score; 1 - R^2 is the part of the quantum dynamics outside the
span of the tested descriptors. The residual (QGF minus classical reconstruction) is
released as the `*_residual` datasets and analysed below.

Code: `src/qgf/reconstruction/` (`python -m qgf.reconstruction.run bases` lists the
families). All commands run from the repository root; outputs go to
`results/reconstruction/...`; small reference tables are in `expected/`. Heavy jobs can
be wrapped with `sbatch experiments/reconstruction/slurm.sh <module> <args>` (CPU only).

## Conventions

- Released correlations are ZZ (= 4 C). Sweeps load them with `target_scale = 1/4`
  (occupation convention). R^2 is scale-invariant; the residual datasets and the
  range profile stay in ZZ.
- Time grid = dataset metadata (`meta["times_ns"]`, linspace(100, 4000, 20) ns). The
  reference instant of the instantaneous tables is time index 7 = 1536 ns.
- Nonlinear probes use 5-fold **graph-level** folds (all pairs of a graph in one fold).

## Extended basis (cumulative stacks)

walk powers -> +RRWP -> +Laplacian eigenpairs -> +ring descriptors -> +heat kernel /
shortest path / effective resistance -> +register distance r and coupling J (classical
ceiling) -> +dTWA classical-spin correlation -> +exact two-atom correlation ->
+mean-field response (extended basis). Probes: in-sample OLS (K = 21, with intercept,
time-averaged) and random forest (K = 21, time index 7).

| Paper item | Command | Output | Expected |
|---|---|---|---|
| Unexplained variance with the extended basis (MUTAG) | `uv run python experiments/reconstruction/run.py ceiling` | `results/reconstruction/ceiling/ceiling_summary.csv`, `ceiling_evolution.pdf` | RF: AFM R^2 0.917 (8.3 %), FM 0.860 (14.0 %) — `expected/ceiling_summary_mutag.csv` |
| Table 18 figure (App. H) | `uv run python experiments/reconstruction/plot_paper_ceiling.py` | `results/paper_figures/reconstruction_ceiling.{pdf,png}` | drawn from `expected/ceiling_summary_mutag.csv` |
| FM walk+RRWP+Laplacian ceiling after the leak fix | from the ceiling run, step `walk+rrwp+laplacian` | idem | FM RF 0.848 (graph folds); AFM 0.915 |

## Initial tables (walk / RRWP / Laplacian)

```
uv run python -m qgf.reconstruction.run sweep --families walk rrwp walk+laplacian rrwp+laplacian --probes ols --times all --intercept --output-dir results/reconstruction/p1/ols_intercept
uv run python -m qgf.reconstruction.run sweep --dataset ZINC_small --families walk rrwp walk+laplacian rrwp+laplacian --probes ols --times all --streaming --no-cache --output-dir results/reconstruction/p1/zinc_ols
uv run python -m qgf.reconstruction.run sweep --families walk rrwp walk+laplacian rrwp+laplacian --probes random_forest --nonlinear-k 3 7 21 --output-dir results/reconstruction/p1/rf_graph
uv run python -m qgf.reconstruction.run sweep ... --probes kernel_ridge ...           # idem
uv run python -m qgf.reconstruction.run per-graph --families walk rrwp --k-grid 50 --output-dir results/reconstruction/p1/per_graph
uv run python -m qgf.reconstruction.run short-time --max-nodes 12 --output-dir results/reconstruction/p1/short_time
```

| Item | Expected (`expected/`) |
|---|---|
| Table 13, OLS walk AFM K=1/7/21/50 | MUTAG 0.597/0.782/0.825/0.832 (intercept), ZINC 0.586/0.770/0.778/0.779 — `table13_ols_time_averaged.csv` |
| Tables 14/15, RF/KR, K=21 | RF FM walk+Lap 0.847, RRWP+Lap 0.840 (graph folds) — `tables14_15_nonlinear_probes_mutag.csv` |
| Per-graph Cayley-Hamilton ceiling | AFM walk median 0.876, RRWP 0.879, worst RRWP graph #149 — `per_graph_summary_mutag.csv` |
| Short-time test t <= 30 ns | AFM unweighted walk R^2 >= 0.9997 (pipeline test passes) — `short_time_verdict.csv` |

## Residual datasets

```
uv run python -m qgf.reconstruction.residual_dataset --dataset MUTAG
uv run python -m qgf.reconstruction.residual_dataset --dataset ZINC_small        # request ~160 GB RAM (slurm.sh default); ~3 min on a cpu node
uv run python experiments/reconstruction/run.py check-residuals
```

Pair residual = C - OLS(classical ceiling, all times pooled); one-body residual =
onebody - OLS(node ceiling). Regenerated files agree with the released ones up to the
Laplacian-eigenvector gauge (see below): relative RMS deviation 2-6 % on two-body,
<1e-4 (MUTAG) / 0.7 % (ZINC) on one-body — `expected/residual_dataset_check.csv`. The
released files remain the reference for all downstream results.

## Residual analysis (ZINC, released residuals, time index 7)

```
uv run python experiments/reconstruction/run.py retention \
    --residual-results experiments/reconstruction/inputs/retention_residual.csv \
    --full-results experiments/reconstruction/inputs/retention_full.csv
uv run python experiments/reconstruction/run.py residual-analysis            # stride 5 + dTWA test
uv run python experiments/reconstruction/run.py residual-analysis --stride 1 --skip-classical   # full spectrum / centroids
```

| Item | Expected |
|---|---|
| Spectrum (normalised-Laplacian eigenbasis) | energy centroid AFM initial/reconstruction/residual 1.464/1.492/1.284, FM 0.339/0.322/0.379; FM residual concentrated at low lambda, AFM broadband — `residual_spectral_centroid_zinc.csv`, `residual_energy_cdf_zinc.csv` |
| Range (mean |value| vs shortest-path distance, ZZ) | reconstruction AFM 0.245 (d=1) -> 0.002 (d=10); residual flat ~0.034 -> 0.021 (AFM), 0.047 -> 0.033 (FM) — `residual_range_zinc.csv` |
| Rank-1 / sum-rule test | top-mode energy median AFM 0.23 / FM 0.45, uniform overlap 0.01 / 0.65, long-range tail kept after per-graph demeaning 100 % / 83 % — `residual_rank_test_zinc.csv` |
| Residual vs classical dynamics (dTWA), d >= 4 | R^2 ~ 0.000 (AFM), 0.004 (FM) — `residual_classical_test_zinc.csv` |
| Information retention (AFM+FM residual), from stored per-task scores | mean graph 97.4 %, node 82.5 %, edge 42.4 % — `retention_summary_zinc.csv` |
| Beyond-WL characterisation (motif, orbits), MUTAG | R^2(residual) ~ 0 — `residual_beyond_wl_mutag.csv` (`python -m qgf.reconstruction.residual`) |

Retention is computed from stored per-task scores: the public intrinsic port does not
run the benchmark on the standalone ZINC residual dataset (ZINC_residual), so the per-task
intrinsic scores of the original runs (FM_AFM and random, residual dataset and full-matrix
dataset) are shipped in `inputs/retention_{residual,full}.csv` (exported by
`scripts/internal/reconstruction_export_retention_inputs.py`); the random baseline is taken
from the full run. 5 of the 20 tasks (Chemical: logp, tpsa, fraction_csp3,
num_rotatable_bonds, bond_type) are chemical tasks dropped by the intrinsic port; they are in
the stored scores but not in the paper's graph/node/edge figures. The command also accepts
a tree of intrinsic results (`<root>/<pe_type>/<benchmark>/result.json`).

Not ported: the "correlation with graph invariants" numbers (mean |corr| 0.46 / 0.33 /
0.17); no source code for them was found.

## Fig. 2: short-time recovery on a toy dataset

`experiments/reconstruction/fig2.py` (about 3 s, CPU). The two MUTAG graphs with N = 10
(indices 75 and 115), the graph Hamiltonian `J_NN A` (exact diagonalisation from
`|0...0>`, `J_NN = 54.75` rad/us) on 20 times in [10, 400] ns; per time, one pooled
least-squares fit over the 90 pairs `i < j` of `[1, (A^k)_ij]_{k<=K}` (adjacency) or
`[1, (M^k)_ij]_{k<=K}` (RRWP, `M = D^-1 A`), columns scaled by max |x|, in-sample R^2.

```bash
uv run python experiments/reconstruction/fig2.py --out results/reconstruction/fig2
uv run python experiments/reconstruction/fig2.py --fm-delta paper --out results/reconstruction/fig2
```

`expected/fig2_r2_published.csv` holds the curves of the published figure (original run);
the port reproduces all 720 values (pulse x family x K x time) to 1e-8. R^2 at 400 ns:

| | K = 1 | K = 3 | K = 7 | K = 21 |
|---|---|---|---|---|
| AFM, adjacency / RRWP | 0.465 / 0.466 | 0.796 / 0.836 | 0.800 / 0.838 | 0.811 / 0.858 |
| FM, adjacency / RRWP (published) | 0.027 / 0.022 | 0.328 / 0.160 | 0.363 / 0.208 | 0.403 / 0.434 |
| FM, adjacency / RRWP (`--fm-delta paper`) | 0.665 / 0.671 | 0.797 / 0.792 | 0.816 / 0.806 | 0.853 / 0.846 |

Notes. The published FM panel used `delta = J_NN / 2` (= `delta_inf` for these graphs,
`|E| = N`), not the paper's FM rule `delta = 2 delta_inf` (= `J_NN` here); with the paper's
rule the FM correlations stay largely linearly recoverable (R^2 0.5-0.85 after the early
period instead of ~0-0.4), so the caption's "not recovered anymore" holds for AFM-like
dynamics and for FM only at `delta_inf` (`expected/fig2_r2_paper.csv`). The fit omits the
`max(., 1)` floor of the column scaling and uses `rcond = 1e-10`, as the original. The
molecule insets of the figure are not drawn.

## Protocol notes

1. Nonlinear probes are scored on graph-level folds (no pair of a graph on both sides).
2. Time-dependent features (dTWA, two-body) keep their time alignment, also when a design
   is read back from the disk cache.
3. Physical controls (dTWA, exact two-atom correlation) are simulated on the dataset time
   grid (`meta["times_ns"]`).
4. Laplacian eigenvector signs are fixed by the largest-magnitude entry after rounding to
   10 decimals, so exact ties do not depend on the numpy/LAPACK build. Degenerate
   eigenspaces (97/188 MUTAG graphs) remain basis-ambiguous: R^2 moves by <= 1.5e-3 (OLS)
   and within 0.25 cross-fold std (RF, K=21), but the pooled residual itself moves by a
   few % RMS.
5. The Laplacian pseudo-inverse uses an explicit cutoff (numpy's default 1e-15 can invert
   the null space on some builds).
