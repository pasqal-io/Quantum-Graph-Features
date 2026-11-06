"""Aggregate downstream GT trials into the paper tables.

MUTAG (Table 2, geometry table): per fold, average the seeds; report mean and std over
the 10 folds (std ddof=0 as in Table 2; the geometry-baseline table uses ddof=1).
ZINC (Table 3): mean and std (ddof=0) over the 10 seeds.

    uv run python experiments/downstream_gt/analyze.py --results results/downstream_gt

Writes <results>/<dataset>_summary.csv (expected values: expected/<dataset>_summary*.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = {"mutag": "test_f1_macro", "zinc": "test_mae"}


def read_tree(root: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for p in sorted((root / dataset).glob("*/**/results.json")):
        r = json.loads(p.read_text())
        fold = r["fold"] if r["fold"] is not None else -1
        rows.append((dataset, r["config"], fold, r["seed"], r[METRIC[dataset]]))
    return pd.DataFrame(rows, columns=["dataset", "config", "fold", "seed", "value"])


def aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    """Per config: mean, std (ddof 0), std (ddof 1), number of trials."""
    out = []
    for cfg, g in trials.groupby("config", sort=False):
        per_unit = g.groupby("fold")["value"].mean() if (g.fold >= 0).all() else g["value"]
        v = per_unit.to_numpy()
        out.append((cfg, v.mean(), v.std(), v.std(ddof=1) if len(v) > 1 else np.nan, len(g)))
    return pd.DataFrame(out, columns=["config", "mean", "std", "std_ddof1", "n_trials"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results/downstream_gt"))
    p.add_argument("--datasets", nargs="+", default=["mutag", "zinc"])
    args = p.parse_args()
    pd.set_option("display.width", 200)
    for ds in args.datasets:
        mine = read_tree(args.results, ds)
        if mine.empty:
            print(f"[{ds}] no results under {args.results / ds}")
            continue
        table = aggregate(mine)
        table.to_csv(args.results / f"{ds}_summary.csv", index=False, float_format="%.6f")
        print(f"\n[{ds}] {METRIC[ds]}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
