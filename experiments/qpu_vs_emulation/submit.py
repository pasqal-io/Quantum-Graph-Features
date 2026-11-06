"""Submit the MUTAG AFM hardware run to a Pasqal QPU, or fetch its results (App. D.4).

Credentials are read from the environment only: ``PASQAL_USERNAME``,
``PASQAL_PASSWORD``, ``PASQAL_PROJECT_ID``. Batch ids are passed on the command line.

    # validate all sequences against the live device spec, no submission
    uv run --extra generate python experiments/qpu_vs_emulation/submit.py \
        submit --t-emu 715 --dry-run
    # submit (prints the batch id), then fetch once all jobs are done
    uv run --extra generate python experiments/qpu_vs_emulation/submit.py submit --t-emu 715
    uv run --extra generate python experiments/qpu_vs_emulation/submit.py fetch \
        --batch-id <id> --out data/raw/my_qpu_run.npz

Each graph: register rescaled so that 0.3 J_max = 0.99 max_amp, pulse lengthened to
t_emu * scale^6 (floored to 4 ns) and preceded by a delay up to 6 us, shots from
``shots_clt(N)``. Fetched counts are stored in the raw format with a single time key
(the QPU pulse duration of graph 0).
"""

import argparse
import os
import time
import warnings
from pathlib import Path

from qgf.data import load
from qgf.generate.qpu import (
    afm_qpu_sequence,
    qpu_pulse_duration,
    register_rescale_factor,
    shots_clt,
)
from qgf.generate.raw import save_raw
from qgf.generate.sequences import register_from_positions

TERMINAL = {"DONE", "CANCELED", "TIMED_OUT", "ERROR"}


def connect():
    from pasqal_cloud.sdk import SDK

    missing = [
        v
        for v in ("PASQAL_USERNAME", "PASQAL_PASSWORD", "PASQAL_PROJECT_ID")
        if v not in os.environ
    ]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} in the environment")
    return SDK(
        username=os.environ["PASQAL_USERNAME"],
        password=os.environ["PASQAL_PASSWORD"],
        project_id=os.environ["PASQAL_PROJECT_ID"],
    )


def submit(args) -> None:
    from pulser.json.abstract_repr.deserializer import deserialize_device

    sdk = connect()
    specs = sdk.get_device_specs_dict()
    if args.device not in specs:
        raise SystemExit(f"device {args.device!r} not available: {sorted(specs)}")
    device = deserialize_device(specs[args.device])
    graphs, _ = load("mutag_afm")
    jobs, skipped = [], []
    for data in graphs:
        gid = int(data.initial_id.flatten()[0])
        reg = register_from_positions(data.pos)
        try:
            seq = afm_qpu_sequence(reg, device, args.t_emu)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                serialized = seq.to_abstract_repr()
        except Exception as exc:  # noqa: BLE001 - incompatible with the device: skip
            skipped.append((gid, str(exc)))
            continue
        n = int(data.num_nodes)
        scale = register_rescale_factor(reg, device)
        jobs.append({"runs": shots_clt(n), "serialized_sequence": serialized})
        print(
            f"graph {gid:3d} N={n:2d} shots={shots_clt(n)} scale={scale:.5f} "
            f"t_QPU={qpu_pulse_duration(args.t_emu, scale)} ns"
        )
    print(f"{len(jobs)} jobs, {len(skipped)} skipped: {skipped}")
    if args.dry_run or not jobs:
        return
    batch = sdk.create_batch(serialized_sequence=None, jobs=jobs, device_type=args.device)
    print(f"batch id: {batch.id}")


def fetch(args) -> None:
    sdk = connect()
    batch = sdk.get_batch(args.batch_id)
    while sum(batch.jobs_count_per_status.get(s, 0) for s in TERMINAL) < batch.jobs_count:
        print(f"waiting: {dict(batch.jobs_count_per_status)}", flush=True)
        time.sleep(30)
        batch = sdk.get_batch(args.batch_id)
    graphs, _ = load("mutag_afm")
    jobs = list(batch.ordered_jobs)
    if len(jobs) != len(graphs):
        raise SystemExit(f"{len(jobs)} jobs for {len(graphs)} graphs: map them by hand")
    t_key = args.time_key
    records = [{t_key: dict(j.result)} if j.status == "DONE" and j.result else {} for j in jobs]
    failed = [i for i, r in enumerate(records) if not r]
    meta = {"kind": "qpu", "device": args.device, "pulse": "AFM", "failed_jobs": failed}
    save_raw(
        args.out,
        records,
        [int(g.initial_id.flatten()[0]) for g in graphs],
        [int(g.num_nodes) for g in graphs],
        [t_key],
        meta,
    )
    print(f"wrote {args.out} ({len(failed)} failed jobs: {failed}; resubmit them)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--t-emu", type=int, required=True, help="emulation time to reproduce (ns)")
    s.add_argument("--device", required=True, help="QPU device name (Pasqal Cloud SDK)")
    s.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("--batch-id", required=True)
    f.add_argument("--time-key", type=int, required=True, help="QPU pulse duration (ns)")
    f.add_argument("--device", required=True, help="QPU device name (Pasqal Cloud SDK)")
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    submit(args) if args.cmd == "submit" else fetch(args)


if __name__ == "__main__":
    main()
