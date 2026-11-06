"""Per-seed results of the original GT capacity study (Fig. 6) -> expected CSV.

The original run directory holds ``cap_<D>[_rrwp_only|_rrwp_af_stats]/seed_<s>/results.json``
(no suffix = RRWP + AFM). Writes ``experiments/ablations/expected/capacity_trials.csv``.

    uv run python scripts/internal/capacity_reference.py --results-dir <run directory>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "experiments/ablations/expected/capacity_trials.csv"
SUFFIX_CONFIG = {"_rrwp_only": "rrwp", "_rrwp_af_stats": "rrwp_af_stat", "": "rrwp_af"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    rows = []
    for d in sorted(args.results_dir.glob("cap_*")):
        rest = d.name.removeprefix("cap_")
        capacity, _, suffix = rest.partition("_")
        config = SUFFIX_CONFIG[f"_{suffix}" if suffix else ""]
        for f in sorted(d.glob("seed_*/results.json")):
            r = json.loads(f.read_text())
            rows.append(
                {
                    "config": config,
                    "hidden_dim": int(capacity),
                    "seed": int(f.parent.name.removeprefix("seed_")),
                    "test_mae": r["test_mae"],
                    "test_rmse": r["test_rmse"],
                    "best_epoch": r.get("best_epoch"),
                }
            )
    df = pd.DataFrame(rows).sort_values(["config", "hidden_dim", "seed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} trials -> {args.out}")
    print(df.groupby(["config", "hidden_dim"]).test_mae.agg(["mean", "std", "count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
