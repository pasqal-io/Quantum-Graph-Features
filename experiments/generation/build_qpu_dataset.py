"""Rebuild the hardware dataset (HW-AFM) from the four raw QPU runs.

Readout-corrected jackknife correlations of the runs T2, T2.5, T4, T15, ZZ convention:
two-atom readout correction off the diagonal, single-atom correction on the diagonal
(``C_ii = p (1 - p)`` of the readout-corrected occupation).

    # writes data/processed/mutag_hw_afm.pt and updates the manifest
    uv run python experiments/generation/build_qpu_dataset.py --release
    uv run python experiments/generation/build_qpu_dataset.py --out <file>
"""

import argparse
from pathlib import Path

from qgf.data import load
from qgf.data.io import save_dataset
from qgf.data.manifest import write_manifest
from qgf.data.registry import DEFAULT_ROOT, file_name
from qgf.generate.build import attach_features
from qgf.generate.raw import QPU_RUNS, load_raw, raw_path

CORRECTION_NOTE = (
    "Readout-corrected weighted jackknife (eps, eps') = (0.025, 0.1): two-atom readout "
    "correction for C_ij (i != j); the one-body diagonal uses the single-atom readout "
    "correction, C_ii = p(1-p) of the readout-corrected occupation p."
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--release", action="store_true", help="overwrite the released file")
    args = parser.parse_args(argv)
    if args.release == (args.out is not None):
        parser.error("use either --release or --out")
    graphs, meta = load("mutag_hw_afm")  # graphs, positions and metadata; features are rebuilt
    raws = [load_raw(raw_path(args.raw_dir, r)) for r in QPU_RUNS]
    rebuilt = attach_features(graphs, raws, jackknife=True)
    meta = {k: v for k, v in meta.items() if k not in ("readout_diagonal", "correction_note")}
    meta |= {"name": "mutag_hw_afm", "correction_note": CORRECTION_NOTE}
    out = DEFAULT_ROOT / file_name("mutag_hw_afm") if args.release else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out, rebuilt, meta)
    print(f"wrote {out} ({len(rebuilt)} graphs)")
    if args.release:
        write_manifest(out.parent, {out.name: len(rebuilt)})
        print(f"updated {out.parent / 'manifest.json'}")


if __name__ == "__main__":
    main()
