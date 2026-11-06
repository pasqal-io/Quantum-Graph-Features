"""Reference data of the exact non-unit-disk experiment (internal sources only).

``features``: compare generated ``mutag_exact[_nn]_{afm,fm}`` files with the original exact
features. ``trials``: convert the original per-trial CSV of the GNN runs into
``experiments/non_unit_disk/expected/trials.csv``.

The original runs stored one ``g<id>_n<N>_<AFM|FM>[_nn].npz`` per graph and pulse
(``onebody`` = 4 n (1 - n) [T, N], ``twobody`` = 4 C [T, N, N] with a NaN diagonal).
Prints the max |difference| per file (onebody, off-diagonal twobody) and the graph count.

    uv run python scripts/internal/non_unit_disk_reference.py features \\
        --generated results/non_unit_disk/data --reference <dir with the .npz files>
    uv run python scripts/internal/non_unit_disk_reference.py trials --csv <v5_mutag_exact173.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qgf.data.io import load_dataset

TRIALS_OUT = Path(__file__).resolve().parents[2] / "experiments/non_unit_disk/expected/trials.csv"
RENAME = {"rrwp_laplacian_ring_geom": "rrwp_lap_ring_geom"}
FILES = {
    "mutag_exact_afm": "AFM",
    "mutag_exact_fm": "FM",
    "mutag_exact_nn_afm": "AFM_nn",
    "mutag_exact_nn_fm": "FM_nn",
}


def compare(path: Path, reference: Path, tag: str) -> tuple[int, float, float]:
    graphs, _ = load_dataset(path)
    d_ob = d_tb = 0.0
    for g in graphs:
        gid, n = int(g.initial_id), g.num_nodes
        ref = np.load(reference / f"g{gid:04d}_n{n:02d}_{tag}.npz")
        ob, tb = g.onebody_features.numpy(), g.twobody_features.numpy()
        off = ~np.eye(n, dtype=bool)
        d_ob = max(d_ob, float(np.abs(ob - ref["onebody"]).max()))
        d_tb = max(d_tb, float(np.abs(tb[:, off] - ref["twobody"][:, off]).max()))
    return len(graphs), d_ob, d_tb


def trials(csv: Path, out: Path) -> None:
    d = pd.read_csv(csv)
    d["config"] = d.config.str.replace(*next(iter(RENAME.items())), regex=False)
    d = d.rename(columns={"model_type": "model", "test_f1_macro": "value"})
    d = d[["model", "config", "fold", "seed", "value", "test_accuracy", "best_epoch",
           "train_size", "val_size", "test_size", "abs_pe_dim", "rel_pe_dim"]]  # fmt: skip
    d = d.sort_values(["model", "config", "fold", "seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"{len(d)} trials -> {out}")
    print(d.groupby("config")[["abs_pe_dim", "rel_pe_dim"]].first())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--generated", type=Path, required=True)
    f.add_argument("--reference", type=Path, required=True)
    t = sub.add_parser("trials")
    t.add_argument("--csv", type=Path, required=True)
    t.add_argument("--out", type=Path, default=TRIALS_OUT)
    args = p.parse_args(argv)
    if args.cmd == "trials":
        trials(args.csv, args.out)
        return 0
    for name, tag in FILES.items():
        for path in sorted(args.generated.glob(f"{name}*.pt")):
            stem = path.stem.removeprefix(name)
            if stem and not stem[1].isdigit():  # mutag_exact_fm* also matches nn files
                continue
            n, d_ob, d_tb = compare(path, args.reference, tag)
            print(f"{path.name}: {n} graphs, max|d onebody| {d_ob:.2e}, max|d twobody| {d_tb:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
