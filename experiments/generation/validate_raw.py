"""Recompute the released features from the raw bitstring counts and compare.

Emulation datasets: plug-in connected correlations of the 1000 shots per time.
``mutag_hw_afm``: readout-corrected jackknife of the four QPU runs (single-atom
correction on the diagonal). Everything is scaled to the ZZ convention (x4). Prints, per file,
the max absolute difference with ``data/processed`` (float32 storage).

    uv run python experiments/generation/validate_raw.py --raw-dir data/raw
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qgf.data import load
from qgf.generate.build import raw_features
from qgf.generate.raw import QPU_RUNS, load_raw, raw_path

CHECKS = {
    "mutag_afm": (None, ["mutag_afm"], False),
    "mutag_fm": (None, ["mutag_fm"], False),
    "mutag_hw_afm": (None, QPU_RUNS, True),
    **{
        f"zinc_{p}_{s}": (s, [f"zinc_{p}_{s}"], False)
        for p in ("afm", "fm")
        for s in ("train", "val", "test")
    },
}


def compare(name: str, raw_dir: Path) -> dict:
    split, raws, jackknife = CHECKS[name]
    graphs, _ = load(name.rsplit("_", 1)[0] if split else name, split)
    loaded = [load_raw(raw_path(raw_dir, r)) for r in raws]
    worst = {"onebody": 0.0, "twobody": 0.0}
    for data in graphs:
        gid = int(data.initial_id.flatten()[0])
        parts = [raw_features(r, r.index_of(gid), jackknife=jackknife) for r in loaded]
        ob = np.concatenate([p[0] for p in parts]).astype(np.float32)
        tb = np.concatenate([p[1] for p in parts]).astype(np.float32)
        worst["onebody"] = max(
            worst["onebody"], float(np.abs(ob - data.onebody_features.numpy()).max())
        )
        worst["twobody"] = max(
            worst["twobody"], float(np.abs(tb - data.twobody_features.numpy()).max())
        )
    return {"graphs": len(graphs), "max_abs_diff": worst}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=sorted(CHECKS))
    parser.add_argument("--out", type=Path, default=Path("results/generation/validate_raw.json"))
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    report = {}
    for name in args.only:
        report[name] = compare(name, args.raw_dir)
        print(name, report[name], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    bad = [n for n, r in report.items() if max(r["max_abs_diff"].values()) > args.tol]
    print("OK" if not bad else f"MISMATCH: {bad}")


if __name__ == "__main__":
    main()
