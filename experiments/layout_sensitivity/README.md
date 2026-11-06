# Placement sensitivity of the quantum features

Four ZINC*_subset test graphs (indices 0, 4, 22, 198; N = 16, 20, 12, 11), AFM pulse.
Gaussian noise on every coordinate with std `sigma = s * d_min` (`d_min` = native minimum
distance), 10 runs per level, registers rescaled to r_NN = 5 um when needed; similarity =
Spearman correlation between the off-diagonal correlations (upper triangle, all 20 times
pooled) of the perturbed and the reference layout.

```bash
# analysis of the stored study (data/raw/layout_sensitivity, 324 cases)
uv run python experiments/layout_sensitivity/analyze.py data/raw/layout_sensitivity
# full re-run, one job per graph (0 4 22 198): noise grid, then Blade re-embeddings
uv run --extra generate python experiments/layout_sensitivity/run.py --graph-idx 4 --out results/layout_sensitivity/rerun
uv run --extra generate python experiments/layout_sensitivity/run.py --graph-idx 4 --reembed --out results/layout_sensitivity/rerun
uv run python experiments/layout_sensitivity/analyze.py results/layout_sensitivity/rerun
# appendix figure (Table 23) from the stored study -> results/paper_figures/layout_noise
uv run python experiments/layout_sensitivity/plot_paper_figure.py
```

| case | reported | stored study | public re-run (Blade) |
|---|---|---|---|
| s = 0.02 (hardware-level jitter, 0.1-0.2 um) | 0.819 +- 0.076 | 0.819 +- 0.073 | 0.813 +- 0.075 |
| s = 0.05 | 0.614 +- 0.133 | 0.607 +- 0.099 | 0.596 +- 0.102 |
| 10 alternative placements | 0.565 +- 0.162 | 0.558 +- 0.247 (post-selected), 0.233 +- 0.287 (raw) | 0.585 +- 0.159 (first 10 Blade runs), 0.719 +- 0.086 (best 10 of 100) |
| node-permutation floor | ~0 | -0.001 | 0.001 |

Public re-run on A100: noise grid 6-11 min per graph, re-embedding (100 Blade candidates + 20 emulations) 12-24 min per graph (`expected/rerun_blade_summary.json`). Blade
(`qoolqit.embedding.Blade`, 5 -> 4 -> 3 -> 2 dimensions, 200 steps per round, QUBO =
adjacency with -1 diagonal, seeded random starts, rescaled to r_NN) replaces the internal
force-based embedder, so the re-embedding rows are not expected to match: the reference
layouts come from the internal embedder, and Blade's best-of-100 layouts (median ELU ~0,
GR 1.30, 38/40 with GR > 1) are more regular than its first 10 runs (ELU 0.013, GR 0.79,
17/40) and than the stored post-selected ones. With 3 -> 2 dimensions and 120 steps Blade
rarely finds unit-disk-like layouts and the similarity drops to 0.23 (raw) / 0.30 (best).

The per-timestep early/late table of the original study is reproduced exactly; the
reported values are matched to within 0.01 in the mean but not exactly in the std (the
stored cases may differ from those behind the reported numbers). Perturbed positions are
reproduced bit for bit from the seeds.

The stored study was emulated with pulse modulation (hardware channel rise time), unlike
the released features; the public re-run uses the feature pipeline (no modulation), so
its correlations differ from the stored ones at early times (up to 0.14 in C at 100 ns),
while the similarity statistics agree within 0.01.
