# Statistics of the downstream sweep (paper App. G)

Fixed champions **RRWP** (classical) vs **RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub>**
(hybrid) over the 12 (architecture, dataset) cells: frequentist sign test, best-of-class
sign test (sensitivity check), Bayesian sign test on the cell-level normalised gaps,
Bayesian signed-rank test on the 300 paired MUTAG trials, hierarchical Bayesian test over
the 6 architectures, ROPE sensitivity and posterior intervals.

Code: `src/qgf/stats/pe_benchmark.py` (port of the internal `stats.py`).

## Inputs

| file | content |
|---|---|
| `inputs/gt_trials.csv` | per-trial GT results of the paper runs (16 feature sets; MUTAG 10 folds x 5 seeds, ZINC 10 seeds) |
| `inputs/other_backbones_mutag_trials.csv` | per-trial MUTAG results of GCN, GAT, GINE, Transformer, GeneralConv (same folds/seeds) |
| hard-coded in the module | ZINC\*<sub>subset</sub> (mean, std) of the five other backbones (Table 3) |

Columns: `arch,dataset,config,fold,seed,value` (value = test macro-F1 for MUTAG, test MAE
for ZINC; fold = -1 for ZINC). To use your own GT runs instead of the stored ones, add
`--gt-mutag-root results/downstream_gt/mutag --gt-zinc-root results/downstream_gt/zinc`.

## Command

```bash
uv sync --extra stats
# httpstan's compiled models need libstdc++ from GCC >= 11 at runtime;
# on RHEL 8 clusters e.g.: module load compilers/gcc/13.4.0
uv run python -m qgf.stats.pe_benchmark \
    --trials experiments/stats/inputs/gt_trials.csv \
             experiments/stats/inputs/other_backbones_mutag_trials.csv \
    --output-dir results/stats
```

Runtime 5-15 min on 4 CPU cores (Stan model compile on first use, 4 hierarchical fits, bootstraps).
`--no-hierarchical` skips the Stan part (a few minutes).

Output: `results/stats/stats_report.md` (+ `stats_summary.json`, `cell_aggregates.csv`,
`bayes_*.pdf/png`). Reference output of this command: `expected/stats_report.md`,
`expected/cell_aggregates.csv`.

## Expected numbers

| item (App. G / Sec. 4.1) | expected (`expected/`) |
|---|---|
| fixed-champion sign test | 10/12, p = 0.0193 |
| best-of-class sign test | 11/11 non-tied, p = 4.9e-4 |
| Bayesian sign test, ROPE ±0.5 pooled SD (P_hyb, P_rope, P_cl) | 0.019, 0.981, 0.000 |
| ... ROPE range ±0.2 / ±0.8: P_rope | 0.967 / 0.997 |
| Cohen's d posterior median [95%] | 0.367 [-0.215, 0.952] |
| signed-rank, 300 MUTAG pairs, ROPE ±0.01 F1 | 0.907, 0.000, 0.093 |
| median within-cell std (MUTAG champions) | 0.0774 |
| pooled mean F1 diff [95%] | 0.0050 [-0.0028, 0.0128] |
| hierarchical, ROPE ±0.01 F1 | 0.240, 0.727, 0.033 |
| hierarchical ROPE ±0.005 / ±0.02 | 0.572 / 0.974 |
| global mu median [95%] | 0.0051 [-0.0088, 0.0187] |
| tau median [95%] | 0.0055 [0.0008, 0.0253] |
| per-arch medians | 0.0018 .. 0.0112 |

Tolerances: cell aggregates, sign tests and the t-posterior intervals are deterministic
(exact). The baycomp tests are Monte-Carlo (unseeded): expect +-0.005 on the sign / signed-rank probabilities and +-0.02 on the
hierarchical ones (bootstrap intervals move by a few 0.01).

## Notes

- The "95% HDI" reported in App. G are equal-tailed 2.5-97.5 % intervals: the original
  tried `arviz.hdi` but arviz was not installed, so the quantile fallback was used. The
  port keeps the quantile interval.
- The mu / tau intervals are computed from baycomp's cached posterior of the *last*
  hierarchical fit, i.e. the ROPE ±0.02 run of the sensitivity sweep (as in the original;
  mu and tau barely depend on the ROPE).
- With this repo's own GT ZINC grid (`--gt-zinc-root results/downstream_gt/zinc`,
  `--no-hierarchical`): sign tests unchanged (10/12, p = 0.0193; 11/11, p = 4.9e-4);
  Bayesian sign test P_rope 0.997 (vs 0.981; the GT ZINC normalised gap drops from 0.0074
  to 0.0025, within noise); the MUTAG-only signed-rank and hierarchical tests do not use
  ZINC. `expected/` stays the output of the default command (stored inputs).
- The GT trials in `inputs/` are the stored per-trial results of the GT sweep (emulated
  feature sets only; same configs as `experiments/downstream_gt/`).

## Re-run of the other backbones (MUTAG)

`inputs/other_backbones_mutag_trials.csv` holds the trials of the paper runs. The exact
re-run of the MUTAG grid (`experiments/downstream_gnns`, pinned environment, bit-identical
trials) gives the same file: converted with

```bash
uv run python experiments/stats/gnn_trials.py results/downstream_gnns_exact/mutag \
    results/stats/other_backbones_mutag_trials_rerun.csv
```

all 4000 trials equal the paper input (max difference 0), so every statistic of App. G is
reproduced exactly (`--gnn-mutag-root <tree>` runs the analysis directly on a re-run tree).
