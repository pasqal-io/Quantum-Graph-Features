"""Compare a ladder run (``results/ladder/summary.csv``) with the original study's tables.

Internal use only: the reference tables are the logistic-regression rows of the original
ladder study (RRWP-2 features, and RRWP-3 features on the same graphs), passed as paths.

    uv run python scripts/internal/ladder_compare_reference.py --k2 <table_K2.md> \
        --k3 <table_K3.md> --summary results/ladder/summary.csv \
        --out experiments/ladder/expected/reference_comparison.csv
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ("accuracy", "f1", "roc_auc")


def parse(path: Path, k: int) -> dict[str, dict[str, str]]:
    """``{family: {metric: "mean ± std", "perm_p": str}}`` of the ``logreg`` rows."""
    rows = {}
    for line in path.read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 6 and c[2] == "logreg":
            fam = c[0].replace("rrwpK", f"rrwp{k}")
            rows[fam] = dict(zip(METRICS, c[3:6])) | {"perm_p": c[6] if len(c) > 6 else ""}
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k2", type=Path, required=True)
    parser.add_argument("--k3", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("results/ladder/summary.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    ref = parse(args.k2, 2) | {f: r for f, r in parse(args.k3, 3).items() if "rrwp3" in f}
    port = pd.read_csv(args.summary).set_index("family")
    out = []
    for fam, r in ref.items():
        if fam not in port.index:
            continue
        row = {"family": fam}
        for m in METRICS:
            mean = port.loc[fam, f"{m}_mean"]
            row[f"{m}_reference"] = r[m]
            row[f"{m}_port"] = f"{mean:.3f} ± {port.loc[fam, f'{m}_std']:.3f}"
            row[f"{m}_abs_delta"] = round(abs(round(mean, 3) - float(r[m].split(" ± ")[0])), 3)
        row["perm_p_reference"] = r["perm_p"]
        row["perm_p_port"] = f"{port.loc[fam, 'perm_p']:.3f}" if "perm_p" in port else ""
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(args.out, index=False)
    deltas = df[[f"{m}_abs_delta" for m in METRICS]].max()
    print(f"{len(df)} families; max |delta| of the means: {deltas.to_dict()}")


if __name__ == "__main__":
    main()
