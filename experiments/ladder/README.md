# Synthetic ladder graphs: pinned vs frustrated (logistic regression)

Code: `src/qgf/synthetic/` (`ladder`, `blind`, `features`, `dataset`, `benchmark`).
Data: `ladder_afm`, `ladder_fm` (`qgf.data.load`).

## Task

Ladders with `m` rungs and diagonal **crossings** (`qgf.synthetic.ladder`). Crossings whose
cell parities agree pin a unique ground state of the blockade (MIS) Hamiltonian
(**pinned**, `y = 1`); conflicting parities leave it degenerate (**frustrated**, `y = 0`).
Labels come from the exact ground-state degeneracy (transfer-matrix DP over rungs).

The dataset is built so that short random walks are **blind**: every open ladder with
9-13 rungs is enumerated and grouped by its pooled RRWP-2 signature (sorted entries of
`M, M^2`, `M = D^-1 A`); only groups holding both labels are kept, and each group is
balanced. Inside a group every pooled RRWP-2 (and ring) feature is bit-identical, so
RRWP-2 sits at chance by construction; partners have the same size and number of
crossings, one crossing has moved by a cell.

- 394 graphs, 98 blind groups, 18-26 atoms, 197 pinned.
- Register: exact unit-disk layout (`unit_disk_positions`, `r_NN = 5.01 um`): square
  cells, a crossing cell is a rhombus of two equilateral triangles (crossing diagonal at
  `r_NN`, every non-edge at >= `sqrt 2 r_NN`). Blade is not used: the layout is exact by
  construction and is the register the released features were computed on.
- Quench: the paper's AFM and FM pulses on the public device (`qgf.generate.sequences`),
  20 times in [0.1, 4] us; emu-mps (bond dimension 16, dt = 20 ns) **expectation values**
  (no shot noise, unlike the MUTAG / ZINC features). Times are
  `numpy.linspace(100, 4000, 20)` (float, `meta["times_ns"]`).
- Features (`qgf.synthetic.features`): every family pooled over nodes / pairs / edges
  (quantiles + mean), so the classifier sees graph-level, permutation-invariant vectors.
  AFM / FM = occupation `<n_i>(t)` (`occupation`) + connected correlation `C_ij(t)`
  (`twobody_features / meta["zz_scale_applied"]`); RRWP = RRWP-20; Geometry = register
  distances and `1/r^6` couplings; Lap = Laplacian spectra + `|phi_2..phi_5|`; Ring = ring
  membership.
- Model: standardisation + logistic regression, 5 folds grouped by blind group x 3
  repeats = 15 test folds; mean ± std (ddof 0). The folds are stored in the data
  (`cv_fold`): scikit-learn >= 1.8 shuffles `StratifiedGroupKFold` differently from the
  1.7.1 release behind the table, which would move AUCs by up to 0.02.

## Commands

```bash
uv run python experiments/ladder/run.py                     # paper table, ~10 s on CPU
uv run python experiments/ladder/run.py --full --perm 100 --jobs -1 \
    --expected experiments/ladder/expected/summary.csv      # every family + permutation p, ~1 min
```

Outputs `results/ladder/{table.md, summary.csv, per_fold.csv}`. Notebooks:
`notebooks/ladder_pairs.ipynb` (what changes between pinned and frustrated partners) and
`notebooks/reproduce_table.ipynb` (the table from the data, checked against `expected/`);
run them with `uv run --with jupyter jupyter lab`.

## Expected numbers

`expected/table.md` (paper table) and `expected/summary.csv` / `expected/table_full.md`
(all families, permutation p from 100 label shuffles, floor 0.010):

| Feature set | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| RRWP | 0.557 ± 0.048 | 0.559 ± 0.047 | 0.612 ± 0.065 |
| Geometry | 0.503 ± 0.053 | 0.495 ± 0.080 | 0.501 ± 0.063 |
| RRWP+Lap+Ring+Geom | 0.641 ± 0.077 | 0.643 ± 0.091 | 0.680 ± 0.065 |
| FM | 0.622 ± 0.063 | 0.620 ± 0.071 | 0.670 ± 0.065 |
| AFM | 0.847 ± 0.045 | 0.849 ± 0.045 | 0.914 ± 0.026 |
| AFM+FM | 0.875 ± 0.043 | 0.875 ± 0.044 | 0.941 ± 0.022 |
| RRWP+Lap+Ring+Geom+AFM+FM | 0.882 ± 0.043 | 0.884 ± 0.043 | 0.941 ± 0.023 |

Other rows (`expected/table_full.md`): node count, RRWP-2 and Ring 0.500 (blind by
construction, p > 0.4); RRWP-3 0.622; Laplacian 0.657; ideal ground state 1.000.
Logistic regression is deterministic: a run reproduces these to all printed digits
(`tests/synthetic/test_smoke.py::test_paper_table_from_released_data`, `-m slow`).

## Validation against the original study

- **Tables**: `expected/reference_comparison.csv` compares every cell with the stored
  tables of the original runs (17 families: RRWP-2 and RRWP-3 runs on the same graphs).
  ROC-AUC and permutation p are identical in all 17; accuracy and F1 are identical except
  RRWP-20 (0.557 / 0.559 vs 0.558 / 0.561) and RRWP-20+Lap+Ring+Geom (0.641 / 0.643 vs
  0.643 / 0.646): threshold metrics of an ill-conditioned 600-dimensional fit, computed on
  x86 originally and arm64 here (`scripts/internal/ladder_compare_reference.py`).
- **Data**: `scripts/internal/ladder_export.py` checks that the released graphs are the
  group-balanced subset of the simulated RRWP-2-blind pool, that labels and degeneracies
  equal the DP, that registers are unit-disk and that `C_ii = <n>(1 - <n>)` (max
  deviation 2.5e-16). `generate.py pool` re-enumerates the pool of 936 graphs identically
  (same order, crossings, labels, groups, registers).
- **Emulation**: through the public device of `qgf.generate.sequences`, emu-sv reproduces
  the original exact emu-sv runs of the same register to 9e-12 (AFM) and 4e-14 (FM) in
  `<n_i>` (18 atoms). The released emu-mps bond-16 features are an approximation (max
  `|<n_i>|` error 0.08 vs exact on that graph; validated at the level of the table:
  ROC-AUC within 0.03 of exact emu-sv on the 274 graphs of 18-24 atoms); a bond-16 re-run
  with emu-mps 2.9.1 has an error of the same size but is not bit-identical to the
  released 2.7.7 run.
- **Protocol defects checked in the original code**: no shape-based transposes (time-major
  arrays are transposed explicitly; `test_features.py` covers N = T = 20); no early
  stopping (logistic regression); standardisation is fitted on the training folds only
  (inside the pipeline).

## Regenerating the data

```bash
uv run python experiments/ladder/generate.py pool --out results/ladder/gen/pool.pt     # 936 graphs, ~5 s
PULSE=afm sbatch --array=0-39 experiments/ladder/submit.sh                             # emu-mps, CPU
PULSE=fm  sbatch --array=0-39 experiments/ladder/submit.sh
uv run python experiments/ladder/generate.py assemble --pool results/ladder/gen/pool.pt \
    --shards "results/ladder/gen/*_*.pt" --out results/ladder/gen/processed
uv run python experiments/ladder/run.py --data-root results/ladder/gen/processed
```

474 graphs are simulated (RRWP-2-blind up to 24 atoms, RRWP-3-blind at 26 atoms); emu-mps
bond 16 costs ~90-105 s per graph and pulse on 8 CPU cores. `generate.py emulate --exact`
uses emu-sv (dt = 40 ns, identical to dt = 10 ns to 4e-12; <= 24 atoms; ~12 s at 18 atoms,
~145 s at 24 atoms per pulse on an A100). A regenerated dataset gets folds from the
installed scikit-learn and an emu-mps run that is not bit-identical, so expect AUCs within
a few hundredths of the table, not the exact digits.
