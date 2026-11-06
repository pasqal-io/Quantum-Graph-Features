# One-/two-body and model-size ablation on ZINC_tiny (paper Tables 8-10)

Code: `src/qgf/downstream/gnns/ablation.py` (port of the internal `train_quantum_ablation.py`),
model `src/qgf/downstream/gnns/model.py`. Data: `zinc_tiny_{afm,fm}` (`load_splits`).

- **Data.** ZINC_tiny, 886 / 211 / 219 graphs (the dataset of the original runs; the
  released `zinc1500_{afm,fm}` is the same subset minus 13 graphs, with features of the
  paper's protocol). Its quantum features come from an **earlier generation protocol**: one
  constant 6000 ns global pulse with fixed (Omega, delta) = (18, 0) rad/us (AFM-like,
  "antiferro") and (27, 62) rad/us (FM-like, "ferro") for every graph, 1000 shots, 20 times
  (`meta["time_note"]`). The original runs used the occupations `<n_i>(t)` as one-body and
  the connected correlations `C_ij(t)` (diagonal `n(1-n)` included) as two-body features,
  z-scored per time step with training statistics.
- **Feature sets.** `no_pe`; `onebody_only` (1b); `twobody_only` (2b); `onebody_twobody`
  (1b + 2b); "All feat." (`zinc_mode = zinc`) adds the bond-type embedding to the edges,
  "Node feat." (`no_zinc`) does not. Tables 8-10 use "All feat.".
- **Models.** GCN and GAT (`GNNWithPE`, 95 atom types) at the sizes of Table 7 (tiny ...
  giant); training as the ZINC benchmark (AdamW lr 5e-4, wd 1e-5, batch 128, <= 1000 epochs,
  warmup 50, early stopping on validation MAE, patience 200), CPU.
- **Pooling (as in the paper).** Each cell pools the trials of several sweeps made with
  different settings (`SETTINGS`): `v1` (add pooling, no dropout, no LayerNorm, the original
  CLI defaults), `v2_gat` / `v2_gat_16h` (mean pooling, dropout 0.1, LayerNorm, 1 / 16 GAT
  heads), `v2_gcn_shift` / `v2_gcn_clamp` (same, GCN edge weights shifted / clamped). Several
  sweeps re-ran the same seeds: the cells count those trials once per sweep (on CPU 515 of
  the 545 repeats are identical, see below; the re-run runs each distinct trial once). The
  baseline row pools the ferro and antiferro no-PE cells. Table 8 pools all model sizes.

## Commands

```bash
uv run python experiments/ablations/run.py --models GCN --sizes tiny --limit 1    # one trial
NTASKS=8 sbatch --array=0-7 experiments/ablations/submit.sh                       # 1360 trials
uv run python experiments/ablations/analyze.py --out <dir>                         # paper tables
uv run python experiments/ablations/analyze.py --results results/ablations \
    --out experiments/ablations/expected/tables/rerun --csv-out experiments/ablations/expected
```

`run.py` runs every distinct (model, dataset, config, size, zinc_mode, setting, seed) of
`expected/trials.csv` once (1360 trials for "All feat."; `--zinc-modes zinc no_zinc` for
both modes, 2720); `analyze.py` fills the paper's trial list with the re-run values and
writes Tables 8-10; with `--results` it also writes the re-run CSVs (below) and prints the
seed-level and cell-level comparison with the paper and the paper's claims.

## Expected numbers

`expected/trials.csv` (the 4735 original trials with their setting and sweep),
`expected/summary_per_size.csv`, `expected/summary_agg.csv` (identical to the original
summaries) and `expected/tables/paper/` (Tables 8-10 regenerated from them: Table 10 equals
the paper cell for cell, incl. best / runner-up markup).

| Table 8 (all sizes pooled) | GCN | GAT |
|---|---|---|
| No PE | 0.520 ± 0.050 | 0.480 ± 0.032 |
| FM<sub>1b</sub> / FM<sub>2b</sub> / FM | 0.512 / 0.478 / 0.493 | 0.473 / 0.456 / 0.482 |
| AFM<sub>1b</sub> / AFM<sub>2b</sub> / AFM | 0.443 / 0.532 / 0.446 | 0.454 / 0.449 / 0.450 |

## Notes on the original runs

- The captions state "mean ± std over 10 seeds"; the cells actually pool 5-35 rows
  (seeds 0-4 of 1-4 sweeps with different settings, repeated seeds included), so the std
  mixes settings and counts identical repeats several times.
- The features of these tables follow the earlier protocol above, not the AFM / FM
  protocol of the paper (Sec. 2.3); the released `zinc1500_{afm,fm}` hold the latter.
- Of the 545 trials that the original sweeps repeated with the same setting and seed, 30
  (GCN `v1`, antiferro: no-PE very_huge / massive, one-body tiny ... large) gave different
  values across sweeps (up to 0.06 MAE); the others are identical.

## Protocol check (original code and port)

The original `train_quantum_ablation.py` (every committed version, from the first ablation
commit to the v2 sweeps) was checked for the three protocol defects:

- **Shape-based transpose: absent.** One-body `[M, N]` -> `[N, M]` with `.T` and two-body
  `[M, N, N]` -> `[N, N, M]` with `.permute(1, 2, 0)`, unconditionally (the time axis comes
  first in the source arrays); the model and collate functions contain no shape test.
- **Early stopping on the test fold: absent.** Early stopping and the checkpoint use the
  validation MAE (patience 200); the test split is evaluated once, with the best-validation
  weights.
- **Normalisation on the full dataset: absent.** Per-time-step z-score statistics are
  computed on the training split only and applied to validation and test; the dataset
  loader does no normalisation of its own.

The port keeps the same protocol: `normalisation()` uses `splits["train"]` only (tested),
explicit `.T` / `.permute(1, 2, 0)` transposes, and `benchmark.fit` stops on the validation
MAE and evaluates the test split once at the best epoch.

## Exact re-run with this repository

All 1360 distinct "All feat." trials were re-run with the pinned environment of the original
sweeps (`reproduce/paper_env`), four threads per process (as every original sweep), on AMD
EPYC 7742 nodes without GPU:

```bash
ABL_ENV=paper NTASKS=8 PARALLEL=16 sbatch -p internal -c 64 --array=0-7 \
    experiments/ablations/submit.sh --out results/ablations_exact
uv run python experiments/ablations/analyze.py --results results/ablations_exact \
    --out experiments/ablations/expected/tables/rerun --csv-out experiments/ablations/expected
```

- **Trials: 1360 / 1360 bit-identical** to a stored original trial (same MAE, best epoch and
  parameter count; `expected/rerun_trials.csv`).
- **Pooled tables.** The paper pools 2362 rows; 2332 equal the re-run. The other 30 are the
  repeats of the sweep `20260401_140818_GCN_all_sizes` that disagree with the other sweeps
  (GCN `v1`, antiferro): that sweep started at 14:08 and the training code was committed twice
  during it (14:22, 14:43), so its later trials ran other code; the same trials were redone in
  `20260401_213304_GCN_rerun_fixed`, which the re-run reproduces exactly. Consequence for the
  tables (`expected/tables/rerun/`, `expected/rerun_comparison.csv`): **Table 10 is identical**
  cell for cell; Table 8 differs in 2 cells by at most 0.0032 MAE; Table 9 in 3 rows (baseline
  at very huge / massive, AFM<sub>1b</sub> at tiny ... medium, by at most 0.013 MAE; one
  best / runner-up swap at tiny). These 30 rows are left as published.

### Do the paper's conclusions hold?

- **"Quantum features consistently improve over classical features" (Table 8).** 5 of 6
  quantum rows per model are below the baseline (exceptions GCN AFM<sub>2b</sub>, GAT FM);
  best GCN AFM<sub>1b</sub> 0.443, GAT AFM<sub>2b</sub> 0.449. The gains exceed one pooled
  std only for GCN AFM<sub>1b</sub> / AFM; the others (0.004-0.039) are within one std, so
  "consistently" holds for the means and the direction, not as a per-row significant effect.
- **"The large configuration achieves the best performance" (Tables 9-10).** Not literally:
  large is the best size in 1 / 7 GCN rows and 2 / 7 GAT rows, and the row-averaged minimum is
  at huge (GCN 0.482 vs large 0.485, GAT 0.450 vs 0.454). Supported: medium to huge are
  equivalent within noise and larger models (very huge ... giant) do worse, so choosing large
  is reasonable.

## Fig. 6: GT capacity study

`capacity.py`: the GT of Table 3 on ZINC\*<sub>subset</sub> (`zinc_afm`) with RRWP (20 / 20),
RRWP + AFM<sub>stat</sub> (24 / 24) and RRWP + AFM (40 / 40), at hidden width D = 64 / 128 /
256 (4 / 8 / 16 heads, FFN 2D, 4 layers), seeds 0-4; 45 trials on GPU.

```bash
uv run python experiments/ablations/capacity.py run --hidden-dims 64 --seeds 0 --max-epochs 2  # smoke
sbatch --array=0-2 experiments/ablations/submit_capacity.sh
uv run python experiments/ablations/capacity.py plot --results results/ablations/capacity --out results/ablations/capacity
uv run python experiments/ablations/capacity.py plot --out experiments/ablations/expected/capacity  # original runs
```

Original runs (`expected/capacity_trials.csv`, test MAE, mean ± std ddof 0 over 5 seeds):

| | D = 64 | D = 128 | D = 256 |
|---|---|---|---|
| RRWP | 0.1133 ± 0.0047 | 0.1130 ± 0.0019 | 0.1138 ± 0.0012 |
| RRWP + AFM<sub>stat</sub> | 0.1111 ± 0.0038 | 0.1112 ± 0.0067 | 0.1114 ± 0.0033 |
| RRWP + AFM | 0.1214 ± 0.0046 | 0.1138 ± 0.0034 | 0.1107 ± 0.0016 |

Notes on the original runs (a separate internal training script, not the Table 3 runner):

- **Model selection on the test set.** The original trained on train + validation pooled
  (10 421 graphs) and passed the test split as validation set, so early stopping, the
  learning-rate plateau and the checkpoint used test MAE; the reported test errors are
  optimistically biased. The port trains on the training split, selects on the validation
  split and evaluates the test split once (the protocol of Table 3); its numbers are
  therefore not expected to equal the figure. The capacity study was not re-run: the GT
  itself is validated by the Table 2-3 reproduction (`experiments/downstream_gt`).
- **Parameter counts.** The legend's "~500k / ~1M / ~2M params" do not match the models:
  D = 64 / 128 / 256 have 0.15M / 0.58M / 2.27M trainable parameters (RRWP + AFM: 150 869 /
  579 689 / 2 279 057); `results.json` records `num_params`.
