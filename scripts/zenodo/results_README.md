# qgf_results.tar: per-trial and per-cell results of the reference runs

Outputs of the experiments of the `qgf` repository, as produced by the commands below on
the released data (small files only: `.json`, `.csv`, `.parquet`; no logs, caches,
checkpoints or figures). Extract with `tar xf qgf_results.tar` (-> `results/`); the
aggregated tables are also in the repository under `experiments/<item>/expected/`.

| directory | content | command (from the repository root) |
|---|---|---|
| `downstream_gt/{mutag,zinc}/<config>/[fold_k/]seed_s/results.json` | one GT trial (Tables 2-3, geometry table) | `python experiments/downstream_gt/run.py --dataset <mutag\|zinc> --configs paper` |
| `downstream_gt/<dataset>_summary.csv` | per-config mean / std | `python experiments/downstream_gt/analyze.py` |
| `intrinsic/grid/<dataset>/<feature set>/<task>/result.json` | one intrinsic cell (Table 1, strengthened baseline, residual tables) | `experiments/intrinsic/submit_grid.sh` (`--output_dir results/intrinsic/grid`) |
| `reconstruction/mutag/ceiling/` | classical reconstruction ceiling (MUTAG) | `python experiments/reconstruction/run.py ceiling` |
| `reconstruction/{mutag,zinc}/p1/<probe>/` | reconstruction sweeps (OLS, RF, kernel ridge; `results.parquet` per cell) | `python -m qgf.reconstruction.run sweep ...` (see `experiments/reconstruction/README.md`) |
| `reconstruction/residual_analysis/`, `retention/`, `residual_datasets/` | ZINC residual analysis, information retention, regenerated-residual check | `python experiments/reconstruction/run.py <residual-analysis\|retention\|check-residuals>` |
| `reconstruction/residual_beyond_wl/` | beyond-WL characterisation of the residual (MUTAG) | `python -m qgf.reconstruction.residual` |
| `generation/` | raw-count and emulation validation, ZINC subset | `python experiments/generation/{validate_raw,validate_emulation,zinc_subset}.py` |
| `qpu_vs_emulation/fig5.json` | Fig. 5 concordance | `python experiments/qpu_vs_emulation/run.py` |
| `shot_budget/shot_budget.json` | App. D.3 shot budget | `python experiments/shot_budget/run.py` |
| `temporal_pca/pca_{mutag,zinc}.json` | temporal PCA | `python experiments/temporal_pca/run.py` |
| `layout_sensitivity/*summary.json` | placement sensitivity | `python experiments/layout_sensitivity/analyze.py data/raw/layout_sensitivity` |
| `ladder/{summary,per_fold}.csv` | synthetic ladder table (per family / per test fold) | `python experiments/ladder/run.py --full --perm 100` |
| `downstream_gnns/{mutag,zinc}/<model>/<config>/[fold_k/]seed_s/results.json` | one GNN trial (Tables 2-3 GCN / GAT / GINE / Transformer / General columns, geometry and hardware tables); bit-identical to the published trials | `GNN_ENV=paper ... experiments/downstream_gnns/submit.sh` (see its README for threads and CPU) |
| `esa/<fusion>/<config>/fold_k/seed_s/results.json` | one ESA trial (Table 4); bit-identical | `ESA_ENV=paper ... experiments/esa/submit.sh --configs paper` |
| `ablations/**/results.json` | one ZINC<sub>tiny</sub> ablation trial (Tables 8-10); bit-identical | `ABL_ENV=paper ... experiments/ablations/submit.sh` |
| `non_unit_disk/mutag/<model>/<config>/fold_k/seed_s/results.json` | one exact non-unit-disk trial (173 MUTAG graphs); bit-identical | `python experiments/non_unit_disk/run.py` |
| `stats/` | App. G statistics (`stats_summary.json`, `cell_aggregates.csv`) | `python -m qgf.stats.pe_benchmark --trials experiments/stats/inputs/*.csv --output-dir results/stats` |

All commands run with `uv run` (see the repository `reproduce/README.md` for the extras
and the compute used).
