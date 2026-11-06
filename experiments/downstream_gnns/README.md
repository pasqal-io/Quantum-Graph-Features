# Downstream GNN benchmark (paper Tables 2-3, GCN / GAT / GINE / Transformer / General; geometry table)

Code: `src/qgf/downstream/gnns/` (port of the original internal GNN of the PE dataloader
template). Data, folds and encodings are shared with the GT port
(`src/qgf/downstream/gt/`, see `experiments/downstream_gt/README.md`).

- Backbones: PyG `GCNConv`, `GATConv`, `GINEConv`, `TransformerConv`, `GeneralConv`.
  Size `large` for all: 4 message-passing layers of width 64, 4-layer MLP head of width
  64, mean pooling, dropout 0.1, LayerNorm after every layer, one attention head.
- Feature injection (App. E.2): one-body encodings concatenated to the atom embedding
  (dim 75); two-body encodings concatenated to the bond embedding (dim 50) as edge
  attributes (GAT, GINE, Transformer, General), or reduced with the bond embedding to one
  learned scalar edge weight per edge (GCN; weights clamped to [1e-8, 100] then shifted per
  graph so that the smallest is 1e-3).
- MUTAG: 10-fold CV stratified by (label, size quartile), `random_state=42`; fold k test,
  fold k+1 validation; seeds 1-5. AdamW lr 5e-4, weight decay 1e-3, batch 32, 300 epochs,
  20 warm-up epochs, early stopping on validation macro-F1 (patience 75).
- ZINC\*<sub>subset</sub>: fixed split 9475/946/947; seeds 0-4. AdamW lr 5e-4, weight decay
  1e-5, batch 128, 1000 epochs, 50 warm-up epochs, L1 loss, early stopping on validation MAE
  (patience 200). ReduceLROnPlateau (factor 0.5, patience 50) after warm-up, both datasets.
- Reported cell: mean ± standard deviation (ddof 1) over all trials (50 on MUTAG, 5 on
  ZINC), as in the paper tables.

## Commands

```bash
# one trial (CPU, a few minutes on MUTAG)
uv run python experiments/downstream_gnns/run.py --dataset mutag --models GCN --configs rrwp \
    --folds 0 --seeds 1
# full grids on Slurm (finished trials are skipped, so re-submitting resumes)
sbatch --array=0-39 experiments/downstream_gnns/submit.sh --dataset mutag
NTASKS=15 sbatch --array=0-14 experiments/downstream_gnns/submit.sh --dataset zinc  # ~7 h
# tables + comparison with the reference runs
uv run python experiments/downstream_gnns/analyze.py --results results/downstream_gnns \
    --expected experiments/downstream_gnns/expected
```

### Exact reproduction (bit-identical trials)

The published trials are reproduced bit for bit with the pinned environment of the original
runs (`reproduce/paper_env`, `uv sync --frozen --project reproduce/paper_env` once), on
CPU only, on AMD EPYC 7742 nodes (here: `-p internal` without `--gres`; the AVX-512 EPYC 9654
nodes of `-p cpu` give different last bits), with the thread count of the original process:

| trials | threads per process |
|---|---|
| MUTAG, all | 1 |
| ZINC, v2 run (all rows except below) | 4 |
| ZINC, Laplacian, RRWP+Laplacian, geometry rows; and the 30 v2 trials the original resubmitted | 1 |

The 30 resubmitted v2 trials (`resubmit_missing_v2` of the original runs; they replaced the
4-thread originals): GAT RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub> seeds 1-4; GCN and
Transformer AFM (GCN seed 3, Transformer 3-4), FM (0-1), RRWP+AFM (GCN 2, Transformer 1-2),
RRWP+AFM+FM (0-3), RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub> (1-4).

```bash
GNN_ENV=paper NTASKS=2 PARALLEL=64 sbatch -p internal -c 64 --array=0-1 \
    experiments/downstream_gnns/submit.sh --dataset mutag --out results/downstream_gnns_exact
GNN_ENV=paper THREADS=4 NTASKS=15 PARALLEL=30 sbatch -p internal -c 120 --array=0-14 \
    experiments/downstream_gnns/submit.sh --dataset zinc --out results/downstream_gnns_exact
GNN_ENV=paper THREADS=1 NTASKS=1 PARALLEL=100 sbatch -p internal -c 100 --array=0 \
    experiments/downstream_gnns/submit.sh --dataset zinc --out results/downstream_gnns_exact_t1 \
    --configs laplacian rrwp_laplacian rrwp_lap_ring_geom rrwp_lap_ring_geom_af_fm
# + the 30 resubmitted trials with THREADS=1 (--models / --configs / --seeds), then take the
#   1-thread result for those trials
```

Grids: MUTAG 22 feature sets (Table 2 incl. Random and the four hardware rows, geometry
table) x 5 backbones x 50 trials = 5500 trials; ZINC 18 feature sets x 5 backbones x 5 seeds
= 450 trials (one trial per process with the commands above; a ZINC trial takes 2-7 h on
one CPU thread of a fully loaded node).

## Expected numbers

`expected/{mutag,zinc}_summary.csv` (per backbone and feature set: mean, std ddof 1, number
of trials, encoding dimensions, source run) and `expected/{mutag,zinc}_trials.csv` (one row
per trial) are the original runs, collected by `scripts/internal/downstream_gnns_reference.py`.
`expected/rerun_<ds>_{summary,claims,trials,comparison}.csv` are the exact re-run with this
repository (`analyze.py` writes `<ds>_summary.csv`, `<ds>_claims.csv`, `<ds>_trials.csv` and the
per-cell comparison with the paper runs `<ds>_comparison.csv`).

| paper item | config | GCN | GAT | GINE | Transformer | General |
|---|---|---|---|---|---|---|
| Table 2, RRWP | `rrwp` | 0.849±0.081 | 0.862±0.078 | 0.861±0.071 | 0.869±0.066 | 0.868±0.080 |
| Table 2, RRWP+AFM+FM | `rrwp_af_fm` | 0.850±0.090 | 0.872±0.077 | 0.867±0.075 | 0.878±0.080 | 0.859±0.079 |
| Table 2, RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub> | `rrwp_af_stat_fm_stat` | 0.860±0.076 | 0.868±0.074 | 0.862±0.078 | 0.871±0.071 | 0.870±0.078 |
| Table 2, RRWP+HW-AFM<sub>stat</sub> | `rrwp_hw_afm_stat` | 0.852±0.084 | 0.853±0.083 | 0.852±0.076 | 0.861±0.078 | 0.877±0.072 |
| Geometry table, RRWP+L+Rings+Geometry | `rrwp_lap_ring_geom` | 0.858±0.081 | 0.855±0.074 | 0.859±0.058 | 0.849±0.070 | 0.849±0.070 |
| Geometry table, + AFM + FM | `rrwp_lap_ring_geom_af_fm` | 0.861±0.079 | 0.874±0.081 | 0.878±0.057 | 0.877±0.073 | 0.853±0.079 |
| Table 3, RRWP | `rrwp` | 0.252±0.006 | 0.223±0.002 | 0.117±0.007 | 0.126±0.006 | 0.148±0.004 |
| Table 3, RRWP+AFM<sub>stat</sub>+FM<sub>stat</sub> | `rrwp_af_stat_fm_stat` | 0.247±0.008 | 0.209±0.004 | 0.116±0.009 | 0.128±0.002 | 0.149±0.008 |

Sources of the reference cells (all with the hyperparameters above):

| cells | original run |
|---|---|
| Tables 2-3, all rows except Laplacian and hardware | v2 multi-model runs (MUTAG `20260405_132642`, ZINC `20260405_*_all_models_v2_tier*`) |
| RRWP+Laplacian (full) | v3 re-run with 4 Laplacian eigenvectors (`20260422_185525`) |
| Laplacian PE | v3 Laplacian-only re-run (`20260429_004920`), see below |
| HW rows | 4-time-step QPU run (`20260430_155033_..._afm_qpu_large_v2_4steps`) |
| geometry table | strong-baseline v4 runs (`20260728_114208` MUTAG, `20260728_114839` ZINC) |

## LaTeX tables (published runs and re-run)

`expected/tables/paper/` (published runs; equal to the paper: Table 2 cell for cell incl.
best / runner-up markup, and the geometry and hardware tables) and
`expected/tables/rerun/` (this repository's re-run; GT column from `experiments/downstream_gt`):
`table2_mutag.tex`, `table3_zinc.tex`, `table_geometry_mutag.tex`, `table_hardware_mutag.tex`.

```bash
uv run python experiments/downstream_gnns/tables.py --source paper --out <dir>
uv run python experiments/downstream_gnns/analyze.py --results results/downstream_gnns
uv run python experiments/downstream_gnns/tables.py --source rerun --gnn-dir results/downstream_gnns \
    --gt-dir experiments/downstream_gt/expected --out <dir>
```

The paper tables take the GT column as published (constants in `tables.py`); the GT port's
own re-run is in `experiments/downstream_gt`.

## Validation of the port

1. **Encodings (exact).** `tests/downstream/test_gnns_reference_features.py` (slow, internal
   reference) compares the port's one- and two-body encodings with those of the original
   GNN pipeline for 15 MUTAG and 4 ZINC feature sets (fold 3, first graphs and every graph
   with N = T = 20): identical (max difference 0); for the hardware sets the two-body block
   is identical (the released one-body block uses the corrected estimator).
2. **Reference tables (exact).** `expected/` reproduces every GNN cell of Tables 2-3 (95 +
   75 cells, mean and std), every cell of the geometry table and the hardware table of the
   paper; Laplacian row: see below.
3. **Re-training (exact).** Every trial of both grids was re-run with the recipe above
   and is bit-identical to the published one (test metric and best epoch): MUTAG 5500 / 5500,
   ZINC 450 / 450 (`expected/rerun_*`; every cell, std, best / runner-up and claim equals the
   paper). Two details of the port were needed for this: bonds are embedded graph by graph when
   two-body features are present (as the original; the order of the embedding-gradient sum
   otherwise changes the last bit), and the thread count / CPU model above.
4. **Project environment.** With the repository's own environment (newer torch / PyG) the
   trials are not bit-identical; statistically they match (MUTAG: every cell within 0.34 std;
   ZINC: mean 0.55 std, consistent with 5-seed noise), but single-cell orderings of near-tied
   rows can change, so use the pinned environment to reproduce the tables.

## Findings on the paper tables

- **Laplacian PE row.** The paper's Laplacian row (e.g. GCN 0.747 on MUTAG, 0.520 on ZINC)
  comes from the v2 run, where the Laplacian block had dimension 1 instead of 4 (the v3
  re-run fixed it for RRWP+Laplacian, the row the paper labels "(full)", but the
  Laplacian-only row was not updated). The port computes 4 eigenvectors; its reference is
  the v3 Laplacian-only run (GCN 0.800 / ZINC 0.478). The v2 values are kept in
  `expected/*_paper_laplacian_v2.csv`.
- **Hardware rows.** They come from the 4-time-step QPU run (`20260430_155033`), not from
  the earlier 3-time-step run `20260429_011040`; with the released `mutag_hw_afm` they are
  reproduced exactly.
- **Standard deviations.** Table 2 mixes two conventions: the regular rows report the std
  over the 50 trials (ddof 1, column `std`), the four hardware rows the std of the 10 fold
  means (ddof 1, column `std_fold`).
- **Geometry table.** The GNN rows (`rrwp_lap_ring_geom*`) used other ring and geometry
  encodings than the GT row (`rrwp_lap_ring_geo*`): cycle-basis ring membership left
  unscaled, and 5 one-body / 3 two-body register channels (`[sum r^-6, max r^-6, min r,
  mean r, distance to centroid]`, `[r, log r, r^-6]`) instead of the GT's
  `(C6/r^6, r)`; both are ported.
- **ZINC seeds.** Table 3's caption says 10 seeds; the GNN columns used 5 (0-4).
- **Protocol** of the original code (checked): early stopping on the validation fold
  (never the test fold), encoding statistics fitted on the training split only, and no
  shape-based transposition (the encodings are the GT port's, checked bit for bit).
