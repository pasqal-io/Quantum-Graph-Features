"""Ladder graphs as qgf datasets (``ladder_afm``, ``ladder_fm``) and back to benchmark inputs.

Per graph (all tensors; the two files are index-aligned by ``initial_id``):

``x [N, 1]`` zeros, ``edge_index``, ``y [1]`` (1 = pinned), ``pos [N, 2]`` (um, float64:
the exact register of :func:`qgf.synthetic.ladder.unit_disk_positions`), ``m [1]`` rungs,
``crossings [k]`` crossing cells, ``degeneracy [1]``, ``initial_id [1]`` (index in the
enumerated pool), ``blind_group [1]`` / ``blind_group_k3 [1]`` (RRWP-2 / RRWP-3 blind
group, -1 = none), ``cv_fold [R]`` (test fold of the graph in each CV repeat),
``occupation [T, N]`` (``<n_i>(t)``), ``onebody_features [T, N]`` and
``twobody_features [T, N, N]`` (ZZ convention ``4 C``, zero diagonal; float64).

Quench features are expectation values of the emulated state (no shot noise).
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from qgf.generate.correlations import ZZ_SCALE, split_features
from qgf.synthetic.features import add_combos, graph_features
from qgf.synthetic.ladder import ladder, unit_disk_positions

# 20 evaluation times of the ladder runs, ns (float, as in the original runs; the MUTAG /
# ZINC runs use the integer grid ``qgf.generate.sequences.EVALUATION_TIMES_NS``)
TIMES_NS = np.linspace(100, 4000, 20)
EMULATOR_MPS = (
    "emu-mps, max bond dimension 16, precision 1e-5, dt = 20 ns, CPU; expectation values of "
    "Occupation and CorrelationMatrix (no shots)"
)


def make_meta(pulse: str, emulator: str, times_ns=TIMES_NS, cv_note: str = "") -> dict:
    """Metadata of ``ladder_<pulse>`` (pulse ``"afm"`` or ``"fm"``)."""
    return {
        "name": f"ladder_{pulse}",
        "split": None,
        "times_ns": [float(t) for t in times_ns],
        "zz_scale_applied": ZZ_SCALE,
        "kind": "emulation",
        "pulse": pulse,
        "emulator": emulator,
        "selection": (
            "RRWP-2-blind ladders with m = 9..13 rungs (18-26 atoms), simulated graphs only, "
            "group-balanced (qgf.synthetic.blind.balanced_subset, seed 0)"
        ),
        "cv_note": cv_note
        or (
            "cv_fold[r] = test fold of StratifiedGroupKFold(5, shuffle=True, random_state=r) "
            "grouped by blind_group"
        ),
    }


def ladder_data(
    m: int,
    crossings,
    *,
    y: int,
    degeneracy: int,
    initial_id: int,
    blind_group: int = -1,
    blind_group_k3: int = -1,
    cv_fold=(),
    occupation: np.ndarray | None = None,
    connected: np.ndarray | None = None,
) -> Data:
    """One ladder as a qgf graph; ``occupation [T, N]`` and ``connected [T, N, N]``
    (``<n_i n_j> - <n_i><n_j>``, time-major) are optional."""
    G = ladder(m, crossings)
    n = 2 * m
    A = nx.to_numpy_array(G, nodelist=range(n))
    src, dst = np.nonzero(A)

    def long(v):
        return torch.tensor(np.atleast_1d(np.asarray(v, dtype=np.int64)))

    data = Data(
        x=torch.zeros((n, 1)),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        y=long(y),
        pos=torch.from_numpy(unit_disk_positions(G)),
        m=long(m),
        crossings=long(sorted(crossings)),
        degeneracy=long(degeneracy),
        initial_id=long(initial_id),
        blind_group=long(blind_group),
        blind_group_k3=long(blind_group_k3),
        cv_fold=long(cv_fold),
    )
    if occupation is not None:
        occupation = np.asarray(occupation, dtype=np.float64)
        connected = np.asarray(connected, dtype=np.float64)
        if occupation.shape != (occupation.shape[0], n):
            raise ValueError(f"occupation must be [T, {n}], got {occupation.shape}")
        if connected.shape != (occupation.shape[0], n, n):
            raise ValueError(f"connected must be [T, {n}, {n}], got {connected.shape}")
        onebody, twobody = split_features(connected, ZZ_SCALE)
        data.occupation = torch.from_numpy(occupation)
        data.onebody_features = torch.from_numpy(onebody)
        data.twobody_features = torch.from_numpy(twobody)
    return data


def quench_arrays(g: Data, zz_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``(occupation [T, N], C [T, N, N])`` in the occupation convention (``4 C / zz_scale``)."""
    return g.occupation.numpy(), g.twobody_features.numpy() / zz_scale


def benchmark_inputs(afm: tuple[list[Data], dict], fm: tuple[list[Data], dict]) -> dict:
    """Pooled features, labels, groups and folds from the loaded ``ladder_afm`` / ``ladder_fm``.

    Returns ``{"F": {family: [G, d]}, "y": [G], "groups": [G], "folds": [G, R],
    "sizes": [G]}``; the two datasets must hold the same graphs in the same order.
    """
    (ga, meta_a), (gf, meta_f) = afm, fm
    if len(ga) != len(gf):
        raise ValueError(f"ladder_afm has {len(ga)} graphs, ladder_fm {len(gf)}")
    rows = []
    for a, f in zip(ga, gf):
        if int(a.initial_id) != int(f.initial_id) or not torch.equal(a.crossings, f.crossings):
            raise ValueError(f"AFM/FM misaligned at initial_id {int(a.initial_id)}")
        rows.append(
            graph_features(
                int(a.m),
                a.crossings.tolist(),
                a.pos.numpy(),
                quench_arrays(a, meta_a["zz_scale_applied"]),
                quench_arrays(f, meta_f["zz_scale_applied"]),
            )
        )
    F = add_combos({k: np.array([r[k] for r in rows]) for k in rows[0]})
    return {
        "F": F,
        "y": np.array([int(g.y) for g in ga]),
        "groups": np.array([int(g.blind_group) for g in ga]),
        "folds": np.stack([g.cv_fold.numpy() for g in ga]),
        "sizes": np.array([g.num_nodes for g in ga]),
    }
