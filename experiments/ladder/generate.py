"""Regenerate the ladder datasets from scratch: enumerate, embed, quench, assemble.

1. ``pool``: enumerate every open ladder with 9-13 rungs, keep the RRWP-2-blind ones whose
   unit-disk register reproduces the graph (936 graphs; ``initial_id`` = index in this pool).
2. ``emulate``: AFM or FM quench (``qgf.generate.sequences.quench_sequence``, public device)
   of a slice of the graphs to simulate, expectation values at 20 times
   (``qgf.generate.emulate.emulate_expectations``). Default: emu-mps for every size, bond
   dimension 16, dt = 20 ns (the released data); ``--exact`` uses emu-sv (<= 24 atoms).
   Graphs simulated: the RRWP-2-blind graphs up to 24 atoms and the RRWP-3-blind graphs
   (26 atoms), 474 in all.
3. ``assemble``: join the AFM / FM shards, keep the group-balanced subset, assign the CV
   folds and write ``ladder_afm.pt`` / ``ladder_fm.pt``.

    uv run python experiments/ladder/generate.py pool --out results/ladder/gen/pool.pt
    uv run --extra generate python experiments/ladder/generate.py emulate \
        --pool results/ladder/gen/pool.pt --pulse afm --shard 0 --nshards 40 --cpu \
        --out results/ladder/gen/afm_000.pt
    uv run python experiments/ladder/generate.py assemble --pool results/ladder/gen/pool.pt \
        --shards "results/ladder/gen/*_*.pt" --out results/ladder/gen/processed

Cost (emu-mps, 8 CPU cores): ~90-105 s per graph and pulse at 22-26 atoms. Folds:
scikit-learn >= 1.8 shuffles ``StratifiedGroupKFold`` differently from the 1.7.1 release
used for the released folds, so a regenerated dataset gets other folds (AUCs move by up to
0.02); the released ``cv_fold`` is the reference.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from qgf.data.io import load_dataset, save_dataset
from qgf.data.registry import file_name
from qgf.synthetic.blind import balanced_subset, blind_pool
from qgf.synthetic.dataset import EMULATOR_MPS, TIMES_NS, ladder_data, make_meta
from qgf.synthetic.ladder import ladder, verify_unit_disk

M_RANGE = (9, 13)
N_FOLDS, N_REPEATS = 5, 3


def build_pool(m_lo: int = M_RANGE[0], m_hi: int = M_RANGE[1]) -> list:
    """RRWP-2-blind, unit-disk ladders (no quench features), in enumeration order."""
    recs, gid = blind_pool(m_lo, m_hi, [2, 3])
    pool = []
    for i in np.nonzero(gid[2] >= 0)[0]:
        r = recs[i]
        if not verify_unit_disk(ladder(r["m"], r["cross"]))[0]:
            continue
        pool.append(
            ladder_data(
                r["m"],
                r["cross"],
                y=r["y"],
                degeneracy=r["degeneracy"],
                initial_id=len(pool),
                blind_group=int(gid[2][i]),
                blind_group_k3=int(gid[3][i]),
            )
        )
    return pool


def to_simulate(pool: list) -> list[int]:
    """Pool indices of the released runs: RRWP-2-blind up to 24 atoms + RRWP-3-blind."""
    return [
        i
        for i, g in enumerate(pool)
        if (int(g.blind_group) >= 0 and g.num_nodes <= 24) or int(g.blind_group_k3) >= 0
    ]


def emulate_graph(g, pulse: str, exact: bool, use_gpu: bool) -> tuple[np.ndarray, np.ndarray]:
    from qgf.generate.emulate import emulate_expectations
    from qgf.generate.sequences import quench_sequence, register_from_positions

    seq = quench_sequence(register_from_positions(g.pos.numpy()), pulse)
    config = {"dt": 40} if exact else {"dt": 20, "sv_max_qubits": 0, "max_bond_dim": 16}
    if exact and g.num_nodes > 24:
        raise ValueError("exact emu-sv runs are limited to 24 atoms here")
    return emulate_expectations(seq, TIMES_NS, use_gpu=use_gpu, **config)


def cmd_pool(args) -> None:
    pool = build_pool()
    sim = to_simulate(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(args.out, pool, {"name": "ladder_pool", "to_simulate": sim})
    print(f"{len(pool)} RRWP-2-blind graphs, {len(sim)} to simulate -> {args.out}")


def cmd_emulate(args) -> None:
    pool, meta = load_dataset(args.pool)
    todo = meta["to_simulate"][args.shard :: args.nshards]
    out = []
    for i in tqdm(todo, desc=f"{args.pulse} shard {args.shard}/{args.nshards}"):
        g = pool[i]
        occ, conn = emulate_graph(g, args.pulse, args.exact, not args.cpu)
        out.append(
            ladder_data(
                int(g.m),
                g.crossings.tolist(),
                y=int(g.y),
                degeneracy=int(g.degeneracy),
                initial_id=int(g.initial_id),
                blind_group=int(g.blind_group),
                blind_group_k3=int(g.blind_group_k3),
                occupation=occ,
                connected=conn,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(args.out, out, {"pulse": args.pulse, "exact": args.exact})
    print(f"{len(out)} graphs -> {args.out}")


def assign_folds(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``[G, R]`` test fold per CV repeat (StratifiedGroupKFold, random_state = repeat)."""
    folds = np.full((len(y), N_REPEATS), -1)
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for k, (_, test) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            folds[test, rep] = k
    return folds


def cmd_assemble(args) -> None:
    pool, _ = load_dataset(args.pool)
    runs: dict[str, dict[int, object]] = {"afm": {}, "fm": {}}
    exact = set()
    for path in sorted(glob.glob(args.shards)):
        graphs, meta = load_dataset(path)
        runs[meta["pulse"]].update({int(g.initial_id): g for g in graphs})
        exact.add(bool(meta["exact"]))
    if len(exact) > 1:
        raise ValueError("shards mix exact and emu-mps runs")
    y = np.array([int(g.y) for g in pool])
    groups = np.array([int(g.blind_group) for g in pool])
    simulated = np.array([i in runs["afm"] and i in runs["fm"] for i in range(len(pool))])
    groups[~simulated] = -1
    sel = balanced_subset(y, groups)
    folds = assign_folds(y[sel], groups[sel])
    emulator = (
        "emu-sv, dt = 40 ns; expectation values (no shots)" if exact == {True} else EMULATOR_MPS
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for pulse, by_id in runs.items():
        graphs = []
        for i, fold in zip(sel, folds):
            g = by_id[int(i)].clone()
            g.cv_fold = torch.from_numpy(fold.astype(np.int64))
            graphs.append(g)
        save_dataset(args.out / file_name(f"ladder_{pulse}"), graphs, make_meta(pulse, emulator))
    print(f"{len(sel)} graphs, {len(set(groups[sel]))} blind groups -> {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="enumerate the RRWP-2-blind ladders")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("emulate", help="quench one shard of the graphs to simulate")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--pulse", choices=["afm", "fm"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--exact", action="store_true", help="emu-sv (<= 24 atoms) instead of emu-mps")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("assemble", help="join shards into ladder_afm.pt / ladder_fm.pt")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--shards", required=True, help="glob of the emulate outputs")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"pool": cmd_pool, "emulate": cmd_emulate, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
