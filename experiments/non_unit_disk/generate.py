"""Exact (noise-free) AFM / FM features of the MUTAG graphs with N <= 24, for two couplings.

- ``geo``: the register couplings ``C6 / r_ij^6`` of the stored embedding (``pos``);
- ``nn``: the graph Hamiltonian ``J_NN A_ij`` (``J_NN = C6 / r_NN^6``), no ``1 / r^6`` tails,
  which no unit-disk register realises; emulated by passing it as interaction matrix.

The pulse follows the paper's rule computed from the coupling used (AFM: ``Omega = 0.3
J_max, delta = 0``; FM: ``Omega = 0.5 J_max, delta = sum_{i != j} J_ij / (2N)``), 4 us
constant quench, emu-sv state vector (exact expectations, no shots), 20 times in
[0.1, 4.0] us. Output: ``mutag_exact[_nn]_{afm,fm}.pt`` (173 graphs, ZZ convention, as the
released MUTAG files) with ``occupation`` [T, N] in addition.

    uv run --extra generate python experiments/non_unit_disk/generate.py \\
        --coupling nn --pulse afm --out results/non_unit_disk/data            # one file
    uv run --extra generate python experiments/non_unit_disk/generate.py \\
        --coupling nn --pulse afm --start 0 --end 4 --out /tmp/x              # 4 graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from qgf.data import DEFAULT_ROOT, load
from qgf.data.io import save_dataset
from qgf.downstream.gt.data import MUTAG_SUBSETS
from qgf.generate.emulate import emulate_expectations
from qgf.generate.sequences import (
    PULSE_DURATION_NS,
    elevated_device,
    graph_couplings,
    interactions,
    quench_sequence,
    register_from_positions,
)

# The original exact runs used the unrounded grid (the released emulations: integer ns).
TIMES_NS = np.linspace(100, PULSE_DURATION_NS, 20)
COUPLINGS = ("geo", "nn")
PULSES = ("afm", "fm")
MAX_NODES = MUTAG_SUBSETS["exact173"]


def dataset_name(coupling: str, pulse: str) -> str:
    return f"mutag_exact_{pulse}" if coupling == "geo" else f"mutag_exact_nn_{pulse}"


def adjacency(g: Data) -> np.ndarray:
    a = np.zeros((g.num_nodes, g.num_nodes))
    a[g.edge_index[0].numpy(), g.edge_index[1].numpy()] = 1.0
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


def couplings_of(g: Data, coupling: str) -> np.ndarray:
    c6 = elevated_device().interaction_coeff
    if coupling == "geo":
        return interactions(register_from_positions(g.pos), c6)[0]
    if coupling == "nn":
        return graph_couplings(adjacency(g), c6)
    raise ValueError(f"unknown coupling {coupling!r}")


def exact_features(
    g: Data, coupling: str, pulse: str, use_gpu: bool = True
) -> tuple[torch.Tensor, ...]:
    """``(occupation [T, N], onebody 4 n (1 - n) [T, N], twobody 4 C [T, N, N])``."""
    j = couplings_of(g, coupling)
    seq = quench_sequence(register_from_positions(g.pos), pulse, couplings=j)
    occ, conn = emulate_expectations(seq, TIMES_NS, use_gpu=use_gpu, interaction_matrix=j)
    tb = 4.0 * conn
    idx = np.arange(g.num_nodes)
    tb[:, idx, idx] = 0.0  # zero diagonal, as every released file
    f = torch.from_numpy
    return f(occ).float(), f(4.0 * occ * (1.0 - occ)).float(), f(tb).float()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coupling", choices=COUPLINGS, required=True)
    p.add_argument("--pulse", choices=PULSES, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args(argv)

    raw, _ = load("mutag_afm", root=args.root)
    graphs = [g for g in raw if g.num_nodes <= MAX_NODES][args.start : args.end]
    out = []
    for g in tqdm(graphs, desc=f"{args.coupling} {args.pulse}"):
        occ, ob, tb = exact_features(g, args.coupling, args.pulse, use_gpu=not args.cpu)
        out.append(
            Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y, pos=g.pos,
                 initial_id=g.initial_id, occupation=occ, onebody_features=ob,
                 twobody_features=tb)
        )  # fmt: skip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    name = dataset_name(args.coupling, args.pulse)
    meta = {
        "name": name,
        "split": None,
        "times_ns": [float(t) for t in TIMES_NS],
        "zz_scale_applied": 1.0,
        "kind": "emulation",
        "time_note": (
            f"exact emu-sv expectations (no shots), {args.coupling} couplings "
            f"({'C6/r^6 of pos' if args.coupling == 'geo' else 'J_NN A, no register'}), "
            f"MUTAG graphs with N <= {MAX_NODES}"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.start == 0 and args.end is None else f"_{args.start}_{args.end}"
    path = args.out / f"{name}{suffix}.pt"
    save_dataset(path, out, meta)
    print(f"wrote {path} ({len(out)} graphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
