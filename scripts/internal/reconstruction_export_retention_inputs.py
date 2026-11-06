"""INTERNAL ONLY: export the stored per-task intrinsic scores used for the retention numbers.

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

The public intrinsic port does not run the benchmark on the standalone ZINC residual
dataset (ZINC_residual), so the retention inputs are shipped as CSVs (columns pe_type,
benchmark, metric, value; pe_types FM_AFM and random) read from the stored result trees:

    uv run python scripts/internal/reconstruction_export_retention_inputs.py \
        <pioneer>/results/PE_benchmark/20260725_183530_atlas/ZINC_residual \
        <pioneer>/results/PE_benchmark/20260724_223540/ZINC_small

Writes experiments/reconstruction/inputs/retention_{residual,full}.csv.
"""

import sys
from pathlib import Path

from qgf.reconstruction.analysis import load_benchmark_results

OUT = Path(__file__).resolve().parents[2] / "experiments/reconstruction/inputs"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, root in (("residual", sys.argv[1]), ("full", sys.argv[2])):
        frame = load_benchmark_results(root)
        frame = frame[frame["pe_type"].isin(["FM_AFM", "random"])]
        frame.to_csv(OUT / f"retention_{name}.csv", index=False)
        print(name, len(frame), "rows")


if __name__ == "__main__":
    main()
