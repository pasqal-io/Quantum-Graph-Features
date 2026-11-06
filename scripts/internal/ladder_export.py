"""Export the synthetic ladder datasets ``ladder_afm`` / ``ladder_fm`` to the public format.

Internal use only (not needed to reproduce results from the released data). Inputs are the
files of the original ladder study, passed as arguments:

- ``--labels``: ``labels.npz`` of the enumerated RRWP-2-blind pool (m = 9..13): ``m``,
  ``crossings``, ``y``, ``degeneracy``, ``group_K2``, ``group_K3``;
- ``--quench``: consolidated AFM/FM emu-mps run (bond dimension 16, dt = 20 ns),
  keys ``<PULSE>_n<N>_{idx,occ,conn}`` (``conn`` = upper triangle incl. diagonal), ``times_ns``;
- ``--folds``: ``sel`` (graph indices of the headline table) and ``folds [G, 3]`` (test fold
  per CV repeat), written with scikit-learn 1.7.1, whose ``StratifiedGroupKFold`` shuffle
  produced the reference table.

Checks before writing: the selection is the group-balanced subset of the RRWP-2-blind graphs
holding both pulses; labels and degeneracies equal the exact ground-state DP; the registers
are unit-disk; correlation matrices are symmetric with diagonal ``<n>(1 - <n>)``.

    uv run python scripts/internal/ladder_export.py --labels <labels.npz> \
        --quench <quench.npz> --folds <folds.npz> --out data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from qgf.data.io import save_dataset
from qgf.data.manifest import write_manifest
from qgf.data.registry import file_name
from qgf.synthetic.blind import balanced_subset
from qgf.synthetic.dataset import ladder_data, make_meta
from qgf.synthetic.ladder import gs_correlations_dp, ladder, parse_crossings, verify_unit_disk

PULSES = {"AFM": "ladder_afm", "FM": "ladder_fm"}
EMULATOR = (
    "emu-mps 2.7.7 (pulser 1.8.0), max bond dimension 16, precision 1e-5, dt = 20 ns, CPU; "
    "expectation values of Occupation and CorrelationMatrix (no shots). Validated against "
    "exact emu-sv on the 18-24-atom graphs (ROC-AUC within 0.03)."
)


def load_quench(path: Path) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """``{pulse: {graph index: (occupation [T, n], connected [T, n, n])}}``."""
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for key in z.files:
        if not key.endswith("_idx"):
            continue
        pulse, ntag, _ = key.split("_")
        n = int(ntag[1:])
        iu = np.triu_indices(n)
        occ, tri = z[f"{pulse}_{ntag}_occ"], z[f"{pulse}_{ntag}_conn"]
        for g, i in enumerate(z[key]):
            conn = np.zeros((tri.shape[1], n, n))
            conn[:, iu[0], iu[1]] = tri[g]
            conn[:, iu[1], iu[0]] = tri[g]
            out.setdefault(pulse, {})[int(i)] = (occ[g], conn)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--quench", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    lab = np.load(args.labels)
    quench = load_quench(args.quench)
    times = np.load(args.quench)["times_ns"]
    fz = np.load(args.folds)
    sel, folds = fz["sel"], fz["folds"]

    n_all = len(lab["y"])
    groups = lab["group_K2"].copy()
    simulated = np.array([i in quench["AFM"] and i in quench["FM"] for i in range(n_all)])
    groups[~simulated] = -1
    if not np.array_equal(balanced_subset(lab["y"], groups), sel):
        raise ValueError("selection is not the group-balanced subset of the simulated graphs")

    graphs: dict[str, list] = {p: [] for p in PULSES}
    worst_diag = 0.0
    for j, i in enumerate(sel):
        m, cr = int(lab["m"][i]), parse_crossings(lab["crossings"][i])
        G = ladder(m, cr)
        _, _, count = gs_correlations_dp(G)
        if count != int(lab["degeneracy"][i]) or int(count == 1) != int(lab["y"][i]):
            raise ValueError(f"graph {i}: stored label/degeneracy differ from the DP")
        if not verify_unit_disk(G)[0]:
            raise ValueError(f"graph {i}: register is not unit-disk")
        for pulse in PULSES:
            occ, conn = quench[pulse][int(i)]
            if not np.array_equal(conn, conn.transpose(0, 2, 1)):
                raise ValueError(f"graph {i} {pulse}: correlation not symmetric")
            diag = np.diagonal(conn, axis1=1, axis2=2)
            worst_diag = max(worst_diag, float(np.abs(diag - occ * (1 - occ)).max()))
            graphs[pulse].append(
                ladder_data(
                    m,
                    cr,
                    y=int(lab["y"][i]),
                    degeneracy=count,
                    initial_id=int(i),
                    blind_group=int(lab["group_K2"][i]),
                    blind_group_k3=int(lab["group_K3"][i]),
                    cv_fold=folds[j],
                    occupation=occ,
                    connected=conn,
                )
            )
    print(f"{len(sel)} graphs; max |C_ii - n(1-n)| = {worst_diag:.2e}")

    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for pulse, name in PULSES.items():
        meta = make_meta(
            pulse.lower(),
            EMULATOR,
            times,
            "cv_fold[r] = test fold of StratifiedGroupKFold(5, shuffle=True, random_state=r) "
            "grouped by blind_group, scikit-learn 1.7.1 (the release behind the reference "
            "table; newer releases shuffle differently).",
        )
        save_dataset(args.out / file_name(name), graphs[pulse], meta)
        entries[file_name(name)] = len(graphs[pulse])
        print(f"wrote {args.out / file_name(name)}")
    write_manifest(args.out, entries)


if __name__ == "__main__":
    main()
