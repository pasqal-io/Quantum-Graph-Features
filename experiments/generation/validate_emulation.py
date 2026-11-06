"""Re-emulate a few MUTAG graphs with the public pipeline and check the stored features.

For each graph and pulse, emu-sv gives exact expectations ``C_exact`` at the 20 times
plus 1000 fresh shots. With ``sigma`` the jackknife standard error of each entry:

- ``z_stored = (C_stored - C_exact) / sigma_stored``: stored shots vs exact dynamics;
- ``z_fresh  = (C_fresh - C_stored) / sqrt(sigma_fresh^2 + sigma_stored^2)``: two
  independent shot samples of the same state.

Both should be ~N(0, 1) if the public code reproduces the original dynamics.

    uv run --extra generate python experiments/generation/validate_emulation.py \
        --raw-dir data/raw --out results/generation/validate_emulation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qgf.data import load
from qgf.generate.correlations import correlation_from_arrays, jackknife_correlation_from_arrays
from qgf.generate.emulate import emulate
from qgf.generate.raw import encode_counts, load_raw, raw_path
from qgf.generate.sequences import EVALUATION_TIMES_NS, quench_sequence, register_from_positions


def pick_graphs(graphs, sizes=(10, 12, 14, 16, 18, 20)) -> list[int]:
    """One graph per target size (closest, lowest index first), as initial ids."""
    n = np.array([int(g.num_nodes) for g in graphs])
    out = []
    for s in sizes:
        for i in np.argsort(np.abs(n - s), kind="stable"):
            gid = int(graphs[i].initial_id.flatten()[0])
            if gid not in out:
                out.append(gid)
                break
    return out


def _bits(counts: dict[str, int]):
    b, c = encode_counts(counts)
    n = len(next(iter(counts)))
    m = (b[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)
    return m.astype(float), c.astype(float)


def _summary(z: np.ndarray) -> dict:
    return {
        "entries": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "frac_abs_gt_3": float(np.mean(np.abs(z) > 3)),
        "max_abs": float(np.abs(z).max()),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--ids", type=int, nargs="*", default=None)
    parser.add_argument("--pulses", nargs="*", default=["afm", "fm"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("results/generation/validate_emulation.json")
    )
    args = parser.parse_args(argv)

    graphs, _ = load("mutag_afm")
    by_id = {int(g.initial_id.flatten()[0]): g for g in graphs}
    ids = args.ids or pick_graphs(graphs)
    report = {"initial_ids": ids, "num_nodes": [int(by_id[i].num_nodes) for i in ids]}
    for pulse in args.pulses:
        raw = load_raw(raw_path(args.raw_dir, f"mutag_{pulse}"))
        z_stored, z_fresh, max_dev = [], [], 0.0
        for gid in ids:
            data, g = by_id[gid], raw.index_of(gid)
            seq = quench_sequence(register_from_positions(data.pos), pulse)
            res = emulate(seq, EVALUATION_TIMES_NS, exact=True, use_gpu=not args.cpu)
            iu = np.triu_indices(int(data.num_nodes))
            for t, t_ns in enumerate(EVALUATION_TIMES_NS):
                occ = res["occupation"][t_ns]
                c_exact = res["correlation"][t_ns] - np.outer(occ, occ)
                b, w = raw.bit_matrix(g, t)
                c_s = correlation_from_arrays(b, w)
                _, se_s = jackknife_correlation_from_arrays(b, w, epsilons=None)
                bf, wf = _bits(res["counts"][t_ns])
                c_f = correlation_from_arrays(bf, wf)
                _, se_f = jackknife_correlation_from_arrays(bf, wf, epsilons=None)
                ok = (se_s[iu] > 1e-9) & (se_f[iu] > 1e-9)
                z_stored.append(((c_s - c_exact)[iu] / se_s[iu])[ok])
                z_fresh.append(((c_f - c_s)[iu] / np.hypot(se_s, se_f)[iu])[ok])
                max_dev = max(max_dev, float(np.abs(c_s - c_exact).max()))
            print(f"{pulse} graph {gid} (N={int(data.num_nodes)}) done", flush=True)
        report[pulse] = {
            "z_stored_vs_exact": _summary(np.concatenate(z_stored)),
            "z_fresh_vs_stored": _summary(np.concatenate(z_fresh)),
            "max_abs_C_stored_minus_exact": max_dev,
        }
        print(json.dumps(report[pulse], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
