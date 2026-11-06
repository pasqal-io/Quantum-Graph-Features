"""Generate quantum graph features (App. D.1) for embedded graphs.

Registers come from the stored ``pos`` of a released dataset (``--dataset``) or of any
dataset file in the qgf format (``--input``). Each graph is emulated with a constant
AFM or FM quench, 1000 shots are sampled at the 20 measurement times, and the counts
are written in the raw format (``qgf.generate.raw``); ``--features`` also writes a
qgf dataset with ``onebody_features``/``twobody_features`` (ZZ convention).

    uv run --extra generate python experiments/generation/run.py \
        --dataset mutag_afm --pulse afm --start 0 --end 4 --out results/generation/mutag_afm_0_4

New graphs: compute positions with Blade (``qoolqit.embedding.algorithms.blade``),
rescale so that the minimum distance is >= 5 um, store them as ``pos`` and pass the
file with ``--input``.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from qgf.data import load
from qgf.data.io import load_dataset, save_dataset
from qgf.generate.build import attach_features
from qgf.generate.emulate import emulate
from qgf.generate.raw import load_raw, save_raw
from qgf.generate.sequences import EVALUATION_TIMES_NS, quench_sequence, register_from_positions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="released dataset name, e.g. mutag_afm, zinc_afm")
    src.add_argument("--input", type=Path, help="dataset file in the qgf format, with pos")
    parser.add_argument("--split", default=None)
    parser.add_argument("--pulse", choices=["afm", "fm"], required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU")
    parser.add_argument("--out", type=Path, required=True, help="output prefix")
    parser.add_argument("--features", action="store_true", help="also write <out>.pt")
    args = parser.parse_args(argv)

    graphs, _ = load(args.dataset, args.split) if args.dataset else load_dataset(args.input)
    graphs = graphs[args.start : args.end]
    records = []
    for data in tqdm(graphs, desc=f"{args.pulse} emulation"):
        seq = quench_sequence(register_from_positions(data.pos), args.pulse)
        records.append(
            emulate(seq, EVALUATION_TIMES_NS, args.shots, use_gpu=not args.cpu)["counts"]
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".npz")
    meta = {"kind": "emulation", "pulse": args.pulse.upper(), "shots": args.shots}
    save_raw(
        raw_path,
        records,
        [int(g.initial_id.flatten()[0]) for g in graphs],
        [int(g.num_nodes) for g in graphs],
        EVALUATION_TIMES_NS,
        meta,
    )
    print(f"wrote {raw_path}")
    if args.features:
        featured = attach_features(graphs, [load_raw(raw_path)])
        meta = {
            "name": args.out.name,
            "split": args.split,
            "times_ns": EVALUATION_TIMES_NS,
            "zz_scale_applied": 4.0,
            "kind": "emulation",
        }
        save_dataset(args.out.with_suffix(".pt"), featured, meta)
        print(f"wrote {args.out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
