# Feature generation (App. D.1)

Code: `src/qgf/generate/` (`sequences`, `emulate`, `correlations`, `raw`, `build`, `qpu`).
Needs the `generate` extra (`uv run --extra generate ...`) for anything that touches pulser
or the emulators; recomputing features from raw counts needs numpy only.

## Pipeline

1. **Register**: atom positions are the stored `pos` of each graph (um, minimum distance
   5.01 um). The original placement code is not part of this repository; for new graphs
   use Blade (see below).
2. **Device**: a public `pulser.devices.VirtualDevice` (`qgf.generate.sequences.virtual_device`)
   built from the paper's parameters only: Rydberg level 60 (`C6/hbar = 8.66e5 um^6 rad/us`),
   `r_NN = 5.01 um`, one global Rydberg channel with maximum amplitude and detuning raised to
   `3 J_NN` (`J_NN = C6 / r_NN^6`), no modulation, 1000 um radial limit. Without
   modulation the dynamics depend only on `C6`, the positions and the pulse; the channel
   limits are only validity checks (checked by re-emulation, see Validation).
3. **Pulse** (constant, 4 us, global Rydberg channel), with `J_max = C6 / r_min^6`:
   - AFM: `Omega = 0.3 J_max`, `delta = 0`;
   - FM: `Omega = 0.5 J_max`, `delta = sum_{i != j} J_ij / (2N)`.
4. **Emulation**: emu-sv for N <= 24, emu-mps above (max bond dimension 1024), `dt = 10 ns`,
   no modulation; 1000 shots at `numpy.linspace(100, 4000, 20, dtype=int)` ns.
5. **Features**: `C = <n_i n_j> - <n_i><n_j>` from the shots, ZZ convention `4 C`;
   `onebody_features = diag` `[T, N]`, `twobody_features` = `C` with zero diagonal `[T, N, N]`.
   QPU data use the readout-corrected weighted jackknife instead (App. D.4).

```bash
uv run --extra generate python experiments/generation/run.py --dataset mutag_afm --pulse afm \
    --start 0 --end 4 --out results/generation/mutag_afm_0_4 --features
```

## Embedding new graphs with Blade

Blade (`qoolqit`, in the `generate` extra) places atoms so that the Rydberg interactions
follow a target matrix; with the graph adjacency as off-diagonal and -1 on the diagonal,
neighbours end up close and non-neighbours far apart:

```python
import numpy as np
from qoolqit.embedding import Blade, BladeConfig


def embed(adj: np.ndarray, seed: int = 0, r_nn: float = 5.0) -> np.ndarray:
    qubo = adj.astype(float).copy()
    np.fill_diagonal(qubo, -1.0)
    start = np.random.default_rng(seed).uniform(size=(len(adj), 5))
    cfg = BladeConfig(dimensions=(5, 4, 3, 2), steps_per_round=200, starting_positions=start)
    coords = Blade(cfg).embed(qubo).coords
    pos = np.array([coords[i] for i in range(len(adj))])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return pos * r_nn / d[np.triu_indices(len(adj), 1)].min()  # minimum distance r_NN
```

Going through 5 -> 4 -> 3 -> 2 dimensions matters: with (3, 2) none of 10 starts gave a
layout with a gap between edges and non-edges (GR > 1) on 11- and 20-atom ZINC graphs;
with (5, 4, 3, 2) 9/10 and 3/10 did (about 6-9 s per embedding).
Blade results depend on the random start: generate several candidates and keep the best
by edge-length uniformity / gap ratio (`experiments/layout_sensitivity/run.py`:
`blade_candidates`, `select_best`). Store the positions as `pos` (float32, um) in a qgf
dataset (`qgf.data.io.save_dataset`) and run `run.py --input <file>`.

## Compute

torch 2.11+cu128 runs on the A100 nodes. One AFM emulation (20 times, 1000 shots),
`experiments/generation/benchmark_emulation.py`:

| N | backend | A100 GPU | CPU (8 cores) |
|---|---|---|---|
| 12 | emu-sv | 9.6 s | 6.3 s |
| 17 | emu-sv | 9.1 s | 40.6 s |
| 20 | emu-sv | 12.6 s | 354 s |
| 24 | emu-sv | 172 s | - |
| 28 | emu-mps (bond dim <= 1024) | > 45 min | - |

Small registers are dominated by fixed overheads; above ~16 atoms use the GPU (`--cpu`
otherwise). MUTAG (N <= 28) fits on one A100; ZINC*_subset needed ~800 GPU-days because
of emu-mps on the largest graphs.

## Raw measurement records

`data/raw/**/*.npz` (Zenodo `qgf_raw.tar`), format documented in `qgf.generate.raw`;
`qgf.generate.raw.raw_path("data/raw", name)` maps a dataset name to its file. Load with
`qgf.generate.raw.load_raw(path)`, then `raw.counts(g, t)` gives
`{bitstring: count}` (character `i` = atom `i`).

| file | content | graphs | times |
|---|---|---|---|
| `emulation/mutag/{afm,fm}.npz` | emulation, 1000 shots | 188 | 20 |
| `emulation/zinc/{afm,fm}_{train,val,test}.npz` | emulation, 1000 shots; AFM holds all 11 494 embedded graphs, FM the 11 368 of ZINC*_subset | 9568/963/963 (AFM), 9475/946/947 (FM) | 20 |
| `qpu/mutag_afm/t2_305ns.npz` | QPU, t_emu = 305 ns (T2), rescaled + time-corrected | 188 | 1 (400 ns pulse) |
| `qpu/mutag_afm/t2p5_512ns.npz` | QPU, 512 ns pulse (T2.5), not rescaled | 188 | 1 |
| `qpu/mutag_afm/t4_715ns.npz` | QPU, t_emu = 715 ns (T4), rescaled + time-corrected, Fig. 5 | 188 | 1 (940 ns pulse) |
| `qpu/mutag_afm/t15_4000ns.npz` | QPU, 4 us pulse (T15), not rescaled | 188 | 1 |
| `embeddings/zinc_{split}.npz` | positions + edges of the 11 494 embedded ZINC graphs | 9568/963/963 | - |
| `layout_sensitivity/afm/*.npz` | correlation tensors of the placement study | 4 graphs | 20 |

QPU files carry the hardware registers (`meta["registers"]`) and requested shots
(`shots_clt(N)`, 734-983; the QPU returned one extra shot for two graphs of the T4 run).
The four QPU files are the runs of `mutag_hw_afm` (`qgf.generate.raw.QPU_RUNS`, time order).

## Validation

| check | command | result |
|---|---|---|
| (a) features from stored raw counts == released datasets (ZZ x4) | `uv run python experiments/generation/validate_raw.py` | max abs diff 0 for all MUTAG/ZINC emulation files and `mutag_hw_afm` (`expected/validate_raw.json`) |
| (b) re-emulation of 6 MUTAG graphs (N = 10-20), AFM and FM, public device, with exact emu-sv expectations | `uv run --extra generate python experiments/generation/validate_emulation.py` | `z = (C_stored - C_exact)/se`: mean -0.006/0.004, std 1.01/1.04; fresh vs stored shots: std 1.00/0.99, 0.2-0.3% beyond 3 sigma (`expected/validate_emulation.json`); identical to 4 decimals to the run with the original device specification |
| ZINC*_subset criterion | `uv run python experiments/generation/zinc_subset.py` | selects exactly the released 11 368 ids (`expected/zinc_subset.json`, with the 126 excluded ids) |

ZINC*_subset: the 11 494 embedded graphs are classified by embedding quality,
`ELU = std/mean` of edge lengths and `GR = 1% quantile of non-edge distances / longest edge`:
C4 `ELU > 0.03`, C3 `ELU > 0.015 and GR < 1.3`, C2 `GR < 1.5`, C1 otherwise. C3 and C4
(93/17/16 graphs in train/val/test) are dropped.

Hardware dataset: `build_qpu_dataset.py --release` rebuilds `mutag_hw_afm` from the four
raw QPU runs (readout-corrected weighted jackknife, (eps, eps') = (0.025, 0.1)). Off-diagonal
entries use the two-atom readout correction; the one-body diagonal uses the single-atom
correction, `C_ii = p(1-p)` of the readout-corrected occupation `p` (recorded in
`meta["correction_note"]`).

Internal export of the raw records: `scripts/internal/generation_export_raw.py` (needs
the internal data manager).
