# Exact non-unit-disk graph Hamiltonian on MUTAG

Does the quantum block help when the dynamics follow the molecular graph exactly, without
the `1/r^6` tails of a unit-disk register (a Hamiltonian current 2D analog hardware cannot
realise)? Four feature sets on the same 173 MUTAG graphs (N <= 24), seven GNN backbones:

| config | quantum block (40 / 40 dims) |
|---|---|
| `rrwp_lap_ring_geom` | none: RRWP + Laplacian + ring + register geometry (35 / 33) |
| `rrwp_lap_ring_geom_af_fm` | released AFM + FM emulation (`mutag_{afm,fm}`, 1000 shots) |
| `rrwp_lap_ring_geom_exact` | exact AFM + FM, register couplings `C6 / r_ij^6` (`mutag_exact_{afm,fm}`) |
| `rrwp_lap_ring_geom_exact_nn` | exact AFM + FM, graph Hamiltonian `J_NN A_ij` (`mutag_exact_nn_{afm,fm}`) |

## Data

`generate.py` (public pipeline, `qgf.generate`): constant 4 us quench with the paper's rule
computed from the coupling used (AFM `Omega = 0.3 J_max, delta = 0`; FM `Omega = 0.5 J_max,
delta = sum_{i != j} J_ij / (2N)`), emu-sv state vector, exact `<n_i>` and `<n_i n_j>` (no
shots) at 20 times in [0.1, 4] us (`linspace(100, 4000, 20)`, not rounded). For `nn` the
matrix `J_NN A` (`J_NN = C6 / r_NN^6`, `r_NN = 5.01 um`) is passed as the emulator's
interaction matrix. The subset is every graph that fits the state-vector emulator
(N <= 24): 173 of 188. Files: ZZ convention, `zz_scale_applied = 1`, plus `occupation`.

```bash
sbatch --array=0-3 experiments/non_unit_disk/submit_generate.sh results/non_unit_disk/data  # GPU, ~2.5 h each
```

Validation of the released files (all 173 graphs of each, generated on A100 GPUs in 1.2 h
for three files and 1.7 h for `nn` FM, against the original exact features of the same
graphs; `scripts/internal/non_unit_disk_reference.py features`):

| file | graphs | max \|delta\| one-body | max \|delta\| two-body (off-diagonal) |
|---|---|---|---|
| `mutag_exact_afm.pt` | 173 | 2.98e-08 | 2.98e-08 |
| `mutag_exact_fm.pt` | 173 | 2.98e-08 | 2.98e-08 |
| `mutag_exact_nn_afm.pt` | 173 | 2.98e-08 | 2.97e-08 |
| `mutag_exact_nn_fm.pt` | 173 | 2.98e-08 | 2.98e-08 |

That is float32 rounding: the public pipeline reproduces the original features. The `nn`
emulation also agrees with dense exact diagonalisation of the same Hamiltonian to 2e-3
(`tests/generate/test_non_unit_disk.py`, the difference is the 10 ns step). Shapes,
conventions and alignment with `mutag_afm` are checked by the slow test
`tests/generate/test_non_unit_disk_benchmark.py::test_released_exact_files`
(`uv run pytest -m slow tests/generate/test_non_unit_disk_benchmark.py`).

## Benchmark

`run.py`: GNN backbones of Table 2 (GCN, GAT, GINE, Transformer, General) plus GATv2 and GEN,
size large, Table 2 training; 10 stratified folds rebuilt on the subset (`subset =
"exact173"`, 137-139 / 17-18 / 17-18 graphs), seeds 1-5: 7 x 4 x 50 = 1400 trials (CPU,
about 35 s each single-threaded; the full grid takes about 5 min on 160 cores).

```bash
uv run python experiments/non_unit_disk/run.py --models GCN --configs rrwp_lap_ring_geom_exact_nn --folds 0 --seeds 1
NTASKS=5 PARALLEL=32 sbatch --array=0-4 experiments/non_unit_disk/submit.sh   # QGF_DATA exported
uv run python experiments/non_unit_disk/analyze.py --out experiments/non_unit_disk/expected      # original
uv run python experiments/non_unit_disk/analyze.py --results results/non_unit_disk/mutag \
    --out experiments/non_unit_disk/expected                                                   # re-run
```

## Expected numbers (original runs, `expected/`)

`expected/trials.csv` (1400 trials), `summary.csv`, `paired.csv`,
`tables/paper/table_mutag_exact173.tex`. Mean macro F1 over 50 trials:

| Backbone | Classical | + emulated | + exact, `C6/r^6` | + exact, `J_NN A` |
|---|---|---|---|---|
| GCN | 0.853 | 0.851 | 0.856 | 0.857 |
| GAT | 0.846 | 0.856 | 0.852 | 0.853 |
| GATv2 | 0.857 | 0.859 | 0.851 | 0.832 |
| GINE | 0.845 | 0.854 | 0.860 | 0.856 |
| GEN | 0.849 | 0.854 | 0.840 | 0.834 |
| Transformer | 0.854 | 0.859 | 0.872 | 0.864 |
| General | 0.850 | 0.845 | 0.864 | 0.865 |

Paired over the 350 (backbone, fold, seed) trials (Wilcoxon signed-rank):

| comparison | mean difference | p |
|---|---|---|
| emulated - classical | +0.0035 | 0.33 |
| exact (`C6/r^6`) - classical | +0.0056 | 0.086 |
| exact (`J_NN A`) - classical | +0.0010 | 0.62 |
| exact (`J_NN A`) - exact (`C6/r^6`) | -0.0047 | 0.22 |
| exact (`C6/r^6`) - emulated | +0.0021 | 0.63 |

Reading: on this subset no quantum block is significantly better than the strong classical
baseline, and the graph Hamiltonian is not better than the register couplings; removing
shot noise moves the mean by +0.002. (On all 188 graphs the original seven-backbone runs
of the emulated block give +0.013, p = 1.5e-3; five of these backbones form the geometry
table of `experiments/downstream_gnns`.) These runs are not in the paper.

## Exact re-run with this repository (`expected/rerun_*`)

All 1400 trials re-run with the pinned environment of the original runs
(`reproduce/paper_env`), one thread per process (as the original), on AMD EPYC 7742 nodes
without GPU, with the generated `mutag_exact[_nn]_{afm,fm}` files:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src uv run --frozen --project reproduce/paper_env \
    python experiments/non_unit_disk/run.py --out results/non_unit_disk_exact/mutag --chunk i/64
uv run python experiments/non_unit_disk/analyze.py --results results/non_unit_disk_exact/mutag \
    --out experiments/non_unit_disk/expected
```

**1400 / 1400 trials are bit-identical to the original runs**, in every config (including the
two exact-feature configs: the 3e-8 float32 difference of the regenerated features does not
change any trial) and for all seven backbones; every cell, paired test and the table equal the
original (`rerun_vs_original.csv`: max |delta| 0). A bug found on the way is fixed
(`qgf.downstream.gt.data.load_mutag`: with `subset="exact173"` the 188-graph emulation files
were not restricted to the subset).

Conclusion: no quantum block (emulated, exact register couplings or exact graph Hamiltonian)
is significantly better than the strong classical baseline on this subset (paired Wilcoxon,
all p >= 0.086; table above).
