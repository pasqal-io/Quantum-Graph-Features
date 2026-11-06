"""Synthetic ladder graphs, pinned vs frustrated: logistic regression on pooled features.

Loads ``ladder_afm`` / ``ladder_fm`` (394 RRWP-2-blind ladders, 18-26 atoms), builds the
pooled feature families (``qgf.synthetic.features``) and cross-validates a logistic
regression on the stored folds (``qgf.synthetic.benchmark``).

    uv run python experiments/ladder/run.py                      # paper table (~20 s)
    uv run python experiments/ladder/run.py --full --perm 100    # every family + permutation p

Writes ``results/ladder/{table.md, summary.csv, per_fold.csv}``; with ``--expected`` prints
the largest difference to a reference ``summary.csv``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qgf.data import DEFAULT_ROOT, load
from qgf.synthetic.benchmark import FULL_TABLE, MAIN_TABLE, METRICS, evaluate, markdown_table
from qgf.synthetic.dataset import benchmark_inputs

EXPECTED = Path(__file__).parent / "expected" / "summary.csv"


def compare(summary: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Per family and metric: ``|delta|`` of the means on the 3-decimal scale of the table."""
    merged = summary.merge(expected, on="family", suffixes=("", "_ref"))
    cols = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    if "perm_p" in summary and "perm_p" in expected:
        cols.append("perm_p")
    diff = {c: (merged[c].round(3) - merged[f"{c}_ref"].round(3)).abs() for c in cols}
    return pd.DataFrame({"family": merged["family"], **diff})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/ladder"))
    parser.add_argument("--full", action="store_true", help="every family, not only the paper rows")
    parser.add_argument("--perm", type=int, default=0, help="label shuffles (paper: 100)")
    parser.add_argument("--jobs", type=int, default=1, help="joblib workers (permutation test)")
    parser.add_argument("--expected", type=Path, default=None, help="reference summary.csv")
    args = parser.parse_args(argv)

    inputs = benchmark_inputs(
        load("ladder_afm", root=args.data_root), load("ladder_fm", root=args.data_root)
    )
    y, groups, sizes = inputs["y"], inputs["groups"], inputs["sizes"]
    print(
        f"{len(y)} graphs ({int(y.sum())} pinned), {len(np.unique(groups))} blind groups, "
        f"{sizes.min()}-{sizes.max()} atoms, {inputs['folds'].shape[1]} CV repeats",
        flush=True,
    )
    labels = FULL_TABLE if args.full else MAIN_TABLE
    families = [fam for fam, _ in labels]
    summary, per_fold = evaluate(inputs["F"], y, inputs["folds"], families, args.perm, args.jobs)

    args.out.mkdir(parents=True, exist_ok=True)
    table = markdown_table(summary, labels)
    (args.out / "table.md").write_text(table)
    summary.to_csv(args.out / "summary.csv", index=False)
    per_fold.to_csv(args.out / "per_fold.csv", index=False)
    print(table)
    print(f"wrote {args.out}/table.md, summary.csv, per_fold.csv")

    if args.expected is not None:
        diff = compare(summary, pd.read_csv(args.expected))
        print(diff.to_string(index=False))
        print(f"max |delta| (3 decimals): {diff.drop(columns='family').to_numpy().max():.3f}")


if __name__ == "__main__":
    main()
