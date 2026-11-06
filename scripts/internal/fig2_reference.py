"""Original Fig. 2 curves (pooled R^2, connected correlations) -> expected CSV.

Reads the ``.npz`` of the original short-time run (keys ``K_values``, ``times_us``,
``R2_{AFM,FM}_conn_{walk,rrwp}_pool`` [K, T]) and writes
``experiments/reconstruction/expected/fig2_r2_published.csv`` in the format of ``fig2.py``.

    uv run python scripts/internal/fig2_reference.py --npz <walk_ceiling_mutag_n10_short.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = (
    Path(__file__).resolve().parents[2]
    / "experiments/reconstruction/expected/fig2_r2_published.csv"
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    d = np.load(args.npz)
    rows = []
    for pulse in ("AFM", "FM"):
        for family, key in (("adjacency", "walk"), ("rrwp", "rrwp")):
            r2 = d[f"R2_{pulse}_conn_{key}_pool"]
            for a, k in enumerate(d["K_values"]):
                for b, t in enumerate(d["times_us"]):
                    rows.append({"pulse": pulse, "family": family, "K": int(k),
                                 "t_ns": float(t) * 1000.0, "r2": float(r2[a, b])})  # fmt: skip
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.8f")
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
