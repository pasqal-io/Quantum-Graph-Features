"""Collect intrinsic-benchmark results into a CSV and compare two result sets.

    # results dir -> CSV (dataset, pe_type, benchmark, metric, raw_value, std_dev)
    uv run python experiments/intrinsic/compare.py csv results/intrinsic/grid out.csv

    # compare a results dir or CSV against a reference CSV: |delta| / std_ref per cell
    uv run python experiments/intrinsic/compare.py diff results/intrinsic/grid \
        experiments/intrinsic/expected/results.csv [--md report.md]
"""

import argparse
import csv
import sys
from pathlib import Path

from qgf.intrinsic.tables import load_results

FIELDS = ["dataset", "pe_type", "benchmark", "metric", "raw_value", "std_dev"]


def read(src: str) -> dict[tuple[str, str, str], dict]:
    if src.endswith(".csv"):
        with open(src) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["raw_value"], r["std_dev"] = float(r["raw_value"]), float(r["std_dev"])
    else:
        rows = load_results(src)
    return {(r["dataset"], r["pe_type"], r["benchmark"]): r for r in rows}


def write_csv(src: str, dst: str) -> None:
    rows = sorted(read(src).values(), key=lambda r: (r["dataset"], r["pe_type"], r["benchmark"]))
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {dst}")


def diff(new_src: str, ref_src: str, md: str | None = None, std_floor: float = 1e-4) -> list:
    """Per common cell: z = |new - ref| / max(std_ref, std_floor * max(|ref|, 1e-3))."""
    new, ref = read(new_src), read(ref_src)
    common = sorted(set(new) & set(ref))
    out = []
    for k in common:
        a, b = new[k]["raw_value"], ref[k]["raw_value"]
        s = max(ref[k]["std_dev"], std_floor * max(abs(b), 1e-3))
        out.append((*k, a, b, ref[k]["std_dev"], abs(a - b) / s))
    missing = sorted(set(ref) - set(new))
    lines = [f"cells compared: {len(out)}; missing in new: {len(missing)}"]
    for ds in sorted({o[0] for o in out}):
        zs = [o[-1] for o in out if o[0] == ds]
        lines.append(
            f"{ds}: max |d|/std = {max(zs):.2f}, > 1 std: {sum(z > 1 for z in zs)}/{len(zs)}, "
            f"> 2 std: {sum(z > 2 for z in zs)}/{len(zs)}"
        )
    worst = sorted(out, key=lambda o: -o[-1])[:15]
    lines += [
        "",
        "| dataset | pe_type | task | new | ref | std_ref | abs(d)/std |",
        "|---" * 7 + "|",
    ]
    lines += [
        f"| {o[0]} | {o[1]} | {o[2]} | {o[3]:.5g} | {o[4]:.5g} | {o[5]:.3g} | {o[6]:.2f} |"
        for o in worst
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if md:
        Path(md).write_text(text)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("csv")
    c.add_argument("src")
    c.add_argument("dst")
    d = sub.add_parser("diff")
    d.add_argument("new")
    d.add_argument("ref")
    d.add_argument("--md")
    args = p.parse_args(argv)
    if args.cmd == "csv":
        write_csv(args.src, args.dst)
    else:
        diff(args.new, args.ref, args.md)


if __name__ == "__main__":
    sys.exit(main())
