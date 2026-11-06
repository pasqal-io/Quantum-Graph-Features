"""INTERNAL ONLY: export stored raw measurement records to the public raw format.

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

Needs Pasqal's ``data_manager`` (run with the quantum-leap venv on the cluster):

    <quantum-leap>/.venv/bin/python scripts/internal/generation_export_raw.py --only mutag_afm

Writes ``qgf.generate.raw.raw_path("data/raw", <name>)`` plus, with
``--zinc-embeddings``, ``data/raw/embeddings/zinc_<split>.npz`` holding the register
positions and edges of all 11 494 embedded ZINC graphs (needed to re-derive the
11 368-graph subset). Cloud batch/job ids are deliberately not exported.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from qgf.generate.raw import raw_path, save_raw

EMU_TIMES = [int(t) for t in np.linspace(100, 4000, 20, dtype=int)]
SPLITS = ("train", "val", "test")
QPU_RUNS = {
    # name: (uri suffix, emulation-equivalent time label of the run)
    "mutag_qpu_afm_t2_305ns": (
        "raw_results_305ns_padded",
        "T2: t_emu = 305 ns, rescaled + time-corrected AFM pulse (400 ns on the QPU)",
    ),
    "mutag_qpu_afm_t2p5_512ns": ("raw_results_512ns", "T2.5: 512 ns AFM pulse, not rescaled"),
    "mutag_qpu_afm_t4_715ns": (
        "raw_results_715ns_padded",
        "T4: t_emu = 715 ns, rescaled + time-corrected AFM pulse (940 ns on the QPU; Fig. 5)",
    ),
    "mutag_qpu_afm_t15_4000ns": ("raw_results_4micro", "T15: 4000 ns AFM pulse, not rescaled"),
}


def _dest(out: Path, name: str) -> Path:
    path = raw_path(out, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _register(seq_str: str) -> np.ndarray:
    return np.array([[q["x"], q["y"]] for q in json.loads(seq_str)["register"]])


def _check_register(elem: dict, pos, what: str) -> None:
    if not np.allclose(_register(elem["sequence_str"]), np.asarray(pos, float), atol=1e-4):
        raise AssertionError(f"{what}: register of the raw record != reference positions")


def _ids(ds) -> list[int]:
    return [int(np.asarray(g.initial_id).flatten()[0]) for g in ds]


def export_emulation(name, project_uri, version, ref_ds, out: Path, dm, meta: dict) -> None:
    n = len(ref_ds)
    records, nodes = [], []
    for i in range(n):
        elem = dm.load_element(collection_uri=project_uri, index=i, version=version)
        if not isinstance(elem, dict):  # ZINC AFM records are pickled QuantumResult objects
            elem = {"state_dicts": elem.state_dicts, "sequence_str": elem.sequence_str}
        _check_register(elem, ref_ds[i].embedded_position, f"{name}[{i}]")
        sd = {int(t): dict(c) for t, c in elem["state_dicts"].items()}
        if sorted(sd) != EMU_TIMES:
            raise AssertionError(f"{name}[{i}]: times {sorted(sd)}")
        records.append(sd)
        nodes.append(len(next(iter(sd[EMU_TIMES[0]]))))
        if i % 500 == 0:
            print(f"{name}: {i}/{n}", flush=True)
    save_raw(_dest(out, name), records, _ids(ref_ds), nodes, EMU_TIMES, meta)
    print(f"wrote {_dest(out, name)} ({n} graphs)", flush=True)


def export_qpu(name, uri, label, mutag, out: Path, dm) -> None:
    elems = dm.load(f"MUTAG/QPU/AFM/{uri}")
    assert len(elems) == len(mutag) and all(elems), name
    (time_key,) = {k for e in elems for k in e["state_dicts"]}
    records = [{int(time_key): dict(e["state_dicts"][time_key])} for e in elems]
    nodes = [len(next(iter(r[int(time_key)]))) for r in records]
    runs = [int(json.loads(e["config_str"])["runs"]) for e in elems]
    # The QPU occasionally returns one shot more than requested.
    meta = {
        "name": name,
        "kind": "qpu",
        "device": "Pasqal Fresnel Canada QPU",
        "pulse": "AFM",
        "run": label,
        "time_note": "times_ns is the QPU pulse duration of the run (time key of the records)",
        "requested_shots": runs,
        "registers": [_register(e["sequence_str"]).round(4).tolist() for e in elems],
    }
    save_raw(_dest(out, name), records, _ids(mutag), nodes, [int(time_key)], meta)
    print(f"wrote {_dest(out, name)}", flush=True)


def export_zinc_embeddings(split: str, out: Path, DataManager) -> None:
    v1 = DataManager(project="results").load(f"ZINC_small/{split}/AFM/qcorr_dataset", version=1)
    pos = [np.asarray(g.embedded_position, dtype=np.float32) for g in v1]
    ei = [np.asarray(g.edge_index, dtype=np.int64) for g in v1]
    np.savez_compressed(
        _dest(out, f"zinc_embeddings_{split}"),
        initial_id=np.array(_ids(v1), dtype=np.int64),
        num_nodes=np.array([len(p) for p in pos], dtype=np.int64),
        pos=np.concatenate(pos),
        num_edges=np.array([e.shape[1] for e in ei], dtype=np.int64),
        edge_index=np.concatenate(ei, axis=1),
    )
    print(f"wrote {_dest(out, f'zinc_embeddings_{split}')} ({len(v1)} graphs)", flush=True)


def main() -> None:
    names = ["mutag_afm", "mutag_fm", *QPU_RUNS]
    names += [f"zinc_{p}_{s}" for p in ("afm", "fm") for s in SPLITS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--only", nargs="*", choices=names, default=[])
    parser.add_argument("--zinc-embeddings", nargs="*", choices=SPLITS, default=[])
    args = parser.parse_args()

    from data_manager import DataManager  # internal package

    args.out.mkdir(parents=True, exist_ok=True)
    dm = DataManager(project="results")
    ql = DataManager(project="ql_datasets")
    base = {"kind": "emulation", "shots": 1000}
    for name in args.only:
        if name in QPU_RUNS:
            export_qpu(name, *QPU_RUNS[name], ql.load("MUTAG"), args.out, dm)
        elif name.startswith("mutag_"):
            pulse = name.split("_")[1].upper()
            version = 1 if pulse == "AFM" else 0
            meta = {**base, "name": name, "pulse": pulse, "dataset": "MUTAG"}
            export_emulation(
                name, f"MUTAG/{pulse}/raw_results", version, ql.load("MUTAG"), args.out, dm, meta
            )
        else:
            _, pulse, split = name.split("_")
            pulse = pulse.upper()
            # AFM was emulated on the 11 494 embedded graphs (index-aligned with AFM qcorr v1);
            # FM on the filtered 11 368 (index-aligned with AFM qcorr v2).
            ref = dm.load(
                f"ZINC_small/{split}/AFM/qcorr_dataset", version=1 if pulse == "AFM" else 2
            )
            meta = {**base, "name": name, "pulse": pulse, "dataset": "ZINC_small", "split": split}
            export_emulation(
                name, f"ZINC_small/{split}/{pulse}/raw_results", 0, ref, args.out, dm, meta
            )
    for split in args.zinc_embeddings:
        export_zinc_embeddings(split, args.out, DataManager)


if __name__ == "__main__":
    main()
