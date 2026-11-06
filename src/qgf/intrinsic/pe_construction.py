"""Feature construction for the intrinsic benchmark.

Ported from the original internal implementation (feature construction and
``compute_rrwp``). The original computed every encoding and
min-max normalised it to [-1, 1] over the *whole* dataset in one pass. Here the
two steps are split so that the normalisation can be fitted on any subset of
graphs (``fit_idx``):

- ``compute_classical_views`` stores the raw classical views (RRWP, Laplacian,
  Ring, Interaction, Distance) on each graph;
- ``normalize_dataset`` returns normalised copies, min/max fitted on
  ``fit_idx`` (``None`` = all graphs, the original behaviour).

The per-column normalisation rules and the PE builders are unchanged.
Classical encodings are kept here (not in a shared module) on purpose: other
experiments ship their own faithful variants.
"""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, to_undirected


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def data_to_graph(data: Data) -> nx.Graph:
    """Convert a PyG Data to an undirected networkx graph (nodes 0..N-1)."""
    return to_networkx(data, to_undirected=True)


# ── quantum features: (onebody, twobody) attribute pairs ──────────────────

QUANTUM_ATTRS: dict[str, tuple[str, str]] = {
    "fm": ("onebody_features", "twobody_features"),
    "afm": ("onebody_features_afm", "twobody_features_afm"),
    "hw_afm": ("onebody_features_hw_afm", "twobody_features_hw_afm"),
    "fmres": ("onebody_features_fmres", "twobody_features_fmres"),
    "afmres": ("onebody_features_afmres", "twobody_features_afmres"),
}
CLASSICAL_GROUPS = ("rrwp", "laplacian", "ring", "interaction", "distance")
SUMMARY_GROUPS = ("fm_summary", "afm_summary")
VIEWS = ("diag", "triu", "rowmean")


def _fit_graphs(dataset: list, fit_idx) -> list:
    return dataset if fit_idx is None else [dataset[i] for i in fit_idx]


def normalize_correlations(
    dataset: list[Data],
    ob_attr: str = "onebody_features",
    tb_attr: str = "twobody_features",
    fit_idx=None,
) -> list[Data]:
    """Normalize onebody and twobody features per timestep to [-1, 1].

    min/max per timestep are taken over the graphs ``fit_idx`` (all if None).
    Returns a new list of (shallow-copied) Data objects.
    """
    T = min(_to_numpy(getattr(d, ob_attr)).shape[1] for d in dataset)
    ob_min = np.full(T, np.inf)
    ob_max = np.full(T, -np.inf)
    tb_min = np.full(T, np.inf)
    tb_max = np.full(T, -np.inf)

    for data in _fit_graphs(dataset, fit_idx):
        ob = _to_numpy(getattr(data, ob_attr))[:, :T]
        tb = _to_numpy(getattr(data, tb_attr))[:, :, :T]
        for t in range(T):
            ob_min[t] = min(ob_min[t], ob[:, t].min())
            ob_max[t] = max(ob_max[t], ob[:, t].max())
            tb_min[t] = min(tb_min[t], tb[:, :, t].min())
            tb_max[t] = max(tb_max[t], tb[:, :, t].max())

    result = []
    for data in dataset:
        new_data = copy.copy(data)
        ob = _to_numpy(getattr(data, ob_attr))[:, :T].copy()
        tb = _to_numpy(getattr(data, tb_attr))[:, :, :T].copy()
        for t in range(T):
            ob_range = ob_max[t] - ob_min[t]
            tb_range = tb_max[t] - tb_min[t]
            if ob_range > 1e-12:
                ob[:, t] = 2.0 * (ob[:, t] - ob_min[t]) / ob_range - 1.0
            else:
                ob[:, t] = 0.0
            if tb_range > 1e-12:
                tb[:, :, t] = 2.0 * (tb[:, :, t] - tb_min[t]) / tb_range - 1.0
            else:
                tb[:, :, t] = 0.0
        setattr(new_data, ob_attr, torch.tensor(ob, dtype=torch.float32))
        setattr(new_data, tb_attr, torch.tensor(tb, dtype=torch.float32))
        result.append(new_data)
    return result


def _normalize_per_step(arrays: list[np.ndarray], fit_idx=None) -> list[np.ndarray]:
    """Normalize a list of arrays per last-axis step to [-1, 1] (min/max over fit_idx)."""
    k = arrays[0].shape[-1]
    step_min = np.full(k, np.inf)
    step_max = np.full(k, -np.inf)
    for arr in _fit_graphs(arrays, fit_idx):
        for t in range(k):
            vals = arr[..., t]
            step_min[t] = min(step_min[t], vals.min())
            step_max[t] = max(step_max[t], vals.max())

    normalized = []
    for arr in arrays:
        out = arr.copy()
        for t in range(k):
            r = step_max[t] - step_min[t]
            if r > 1e-12:
                out[..., t] = 2.0 * (out[..., t] - step_min[t]) / r - 1.0
            else:
                out[..., t] = 0.0
        normalized.append(out)
    return normalized


def _extract_triu(twobody: np.ndarray) -> np.ndarray:
    """Extract upper-triangular entries from (N, N, T) -> (M, T)."""
    N = twobody.shape[0]
    idx = np.triu_indices(N, k=1)
    return twobody[idx[0], idx[1], :]


# ── classical encodings (raw) ─────────────────────────────────────────────


def compute_rrwp(graph: nx.Graph, k: int) -> np.ndarray:
    """Relative Random Walk Probabilities (N, N, k): slice t is the (t+1)-step transition matrix."""
    nodelist = sorted(graph.nodes())
    adj = nx.to_numpy_array(graph, nodelist=nodelist)
    num_nodes = adj.shape[0]

    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    transition_matrix = adj / degree[:, np.newaxis]

    pe = np.zeros((num_nodes, num_nodes, k))
    current_power = np.copy(transition_matrix)
    for i in range(k):
        pe[:, :, i] = current_power
        current_power = current_power @ transition_matrix
    return pe


RRWP_K = 20


def rrwp_views(graph: nx.Graph, k: int = RRWP_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(diag, triu, rowmean) of P^2..P^{k+1}.

    The first walk step (P^1) is dropped because its diagonal is identically
    zero for simple graphs; k+1 steps are computed and steps 2..k+1 kept.
    """
    r = compute_rrwp(graph, k=k + 1)[:, :, 1:]
    N = r.shape[0]
    idx = np.triu_indices(N, k=1)
    diag = np.diagonal(r, axis1=0, axis2=1).T
    triu = np.stack([r[:, :, t][idx] for t in range(k)], axis=1)
    return diag, triu, r.mean(axis=1)


LAPLACIAN_PE_DIM = 4


def _compute_laplacian_pe(data: Data, k: int = LAPLACIAN_PE_DIM) -> tuple[np.ndarray, np.ndarray]:
    """First k non-trivial eigenvectors of the symmetric normalised Laplacian.

    Returns onebody (N, k) eigenvector values and twobody (N, N, k) outer products.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if num_nodes <= 1:
        return np.zeros((num_nodes, k)), np.zeros((num_nodes, num_nodes, k))

    edge_index_ud = to_undirected(edge_index, num_nodes=num_nodes)
    adj = to_dense_adj(edge_index_ud, max_num_nodes=num_nodes).squeeze(0)

    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = deg[nonzero].pow(-0.5)

    laplacian = torch.eye(num_nodes, dtype=adj.dtype) - torch.diag(
        deg_inv_sqrt,
    ) @ adj @ torch.diag(deg_inv_sqrt)
    _, eigenvectors = torch.linalg.eigh(laplacian)
    eigenvectors = eigenvectors.numpy()

    # Skip trivial eigenvector (index 0), take next k; pad if not enough
    available = eigenvectors.shape[1] - 1
    take = min(k, max(available, 0))
    selected = eigenvectors[:, 1 : 1 + take]
    if take < k:
        pad = np.zeros((num_nodes, k - take))
        selected = np.concatenate([selected, pad], axis=1)

    # Sign convention per eigenvector: first nonzero component positive
    for j in range(k):
        col = selected[:, j]
        nonzero_idx = np.nonzero(np.abs(col) > 1e-8)[0]
        if len(nonzero_idx) > 0 and col[nonzero_idx[0]] < 0:
            selected[:, j] = -col

    twobody = np.stack([np.outer(selected[:, j], selected[:, j]) for j in range(k)], axis=-1)
    return selected, twobody


def laplacian_views(data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ob, tb = _compute_laplacian_pe(data)
    return ob, _extract_triu(tb), tb.mean(axis=1)


RING_SIZES: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
RING_PE_DIM = len(RING_SIZES)
_RING_COL = {size: index for index, size in enumerate(RING_SIZES)}


def ring_membership_tensor(graph: nx.Graph) -> np.ndarray:
    """Binary co-membership tensor (N, N, 6).

    ``R[i, j, c] = 1`` iff a cycle of the minimum cycle basis of size exactly
    ``RING_SIZES[c]`` contains both i and j. The diagonal is node ring membership.
    """
    nodes = sorted(graph.nodes())
    index = {node: position for position, node in enumerate(nodes)}
    num_nodes = len(nodes)
    tensor = np.zeros((num_nodes, num_nodes, RING_PE_DIM), dtype=np.float32)
    for cycle in nx.minimum_cycle_basis(graph):
        column = _RING_COL.get(len(cycle))
        if column is None:
            continue
        members = [index[node] for node in cycle]
        for a in members:
            for b in members:
                tensor[a, b, column] = 1.0
    return tensor


def ring_views(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = ring_membership_tensor(graph)
    diag = np.stack([t[i, i, :] for i in range(t.shape[0])], axis=0)
    return diag, _extract_triu(t), t.mean(axis=1)


# pulser C6 coefficient for Rydberg level 70 (rad um^6 / us): a global scale that cancels
# after normalisation; kept for faithfulness to the original features.
C6_PULSER_N70: float = 5420158.53


def _pairwise_control_tensors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, N) interaction C6/r^6 and distance r matrices, zero where r == 0."""
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    positive = distances > 0
    interaction = np.zeros_like(distances)
    with np.errstate(divide="ignore"):
        interaction[positive] = C6_PULSER_N70 / distances[positive] ** 6
    distance = np.where(positive, distances, 0.0)
    return interaction, distance


def control_views(data: Data) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Geometric controls from the atom positions: diag = row sum, triu, rowmean."""
    interaction, distance = _pairwise_control_tensors(_to_numpy(data.pos))
    return {
        prefix: (
            m.sum(axis=1, keepdims=True),
            _extract_triu(m[:, :, np.newaxis]),
            m.mean(axis=1, keepdims=True),
        )
        for prefix, m in (("interaction", interaction), ("distance", distance))
    }


def classical_views(data: Data) -> dict[str, np.ndarray]:
    """All raw classical views of one graph, keyed ``{group}_{view}``."""
    G = data_to_graph(data)
    groups = {
        "rrwp": rrwp_views(G),
        "laplacian": laplacian_views(data),
        "ring": ring_views(G),
        **control_views(data),
    }
    return {f"{g}_{v}": arr for g, arrs in groups.items() for v, arr in zip(VIEWS, arrs)}


def attach(dataset: list[Data], views: list[dict[str, np.ndarray]]) -> None:
    for d, v in zip(dataset, views):
        for key, arr in v.items():
            setattr(d, key, arr)


# ── summary statistics of the (normalised) quantum features ───────────────


def _compute_summary(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean (..., 1) and stat (..., 4) = [min, max, mean, std] along the last axis."""
    mean = arr.mean(axis=-1, keepdims=True)
    stat = np.stack(
        [arr.min(axis=-1), arr.max(axis=-1), arr.mean(axis=-1), arr.std(axis=-1)],
        axis=-1,
    )
    return mean, stat


def compute_and_normalize_summary_stats(
    dataset: list[Data], prefixes=("fm", "afm"), fit_idx=None
) -> None:
    """Mean (1 feat) and stat (4 feat) summaries of normalised FM/AFM, normalised again.

    Sets ``{prefix}_{mean|stat}_{diag|triu|rowmean}`` in place.
    """
    for prefix in prefixes:
        ob_attr, tb_attr = QUANTUM_ATTRS[prefix]
        out: dict[str, list[np.ndarray]] = {}
        for d in dataset:
            ob = _to_numpy(getattr(d, ob_attr))  # (N, T)
            tb = _to_numpy(getattr(d, tb_attr))  # (N, N, T)
            for view, arr in (("diag", ob), ("triu", _extract_triu(tb)), ("rowmean", tb.mean(1))):
                m, s = _compute_summary(arr)
                out.setdefault(f"{prefix}_mean_{view}", []).append(m)
                out.setdefault(f"{prefix}_stat_{view}", []).append(s)
        for key, arrays in out.items():
            for d, arr in zip(dataset, _normalize_per_step(arrays, fit_idx)):
                setattr(d, key, torch.tensor(arr, dtype=torch.float32))


# ── feature groups needed by each PE type ─────────────────────────────────

_GEO = ["rrwp", "laplacian", "ring", "interaction", "distance"]
PE_GROUPS: dict[str, list[str]] = {
    "random": [],
    "Interaction": ["interaction"],
    "Distance": ["distance"],
    "RRWP": ["rrwp"],
    "Laplacian": ["laplacian"],
    "RRWP_Laplacian": ["rrwp", "laplacian"],
    "Ring": ["ring"],
    "RRWP_Laplacian_Ring": ["rrwp", "laplacian", "ring"],
    "RRWP_Laplacian_Ring_Quantum": ["rrwp", "laplacian", "ring", "fm", "afm"],
    "RRWP_Laplacian_Ring_Geometric": _GEO,
    "RRWP_Laplacian_Ring_Geometric_Quantum": _GEO + ["fm", "afm"],
    "RRWP_Laplacian_Ring_Geometric_FMres": _GEO + ["fmres"],
    "RRWP_Laplacian_Ring_Geometric_AFMres": _GEO + ["afmres"],
    "RRWP_Laplacian_Ring_Geometric_FMres_AFMres": _GEO + ["fmres", "afmres"],
    "FM_mean": ["fm", "fm_summary"],
    "FM_stat": ["fm", "fm_summary"],
    "FM": ["fm"],
    "AFM_mean": ["afm", "afm_summary"],
    "AFM_stat": ["afm", "afm_summary"],
    "AFM": ["afm"],
    "FM_AFM": ["fm", "afm"],
    "RRWP_FM": ["rrwp", "fm"],
    "RRWP_AFM": ["rrwp", "afm"],
    "RRWP_FM_AFM": ["rrwp", "fm", "afm"],
    "HW_AFM": ["hw_afm"],
    "RRWP_HW_AFM": ["rrwp", "hw_afm"],
}
PE_TYPES = list(PE_GROUPS)


def normalize_dataset(dataset: list[Data], groups, fit_idx=None) -> list[Data]:
    """Normalised shallow copies of ``dataset`` for the given feature groups.

    Every column is min-max scaled to [-1, 1] with min/max taken over the graphs
    ``fit_idx`` (``None`` = whole dataset, as in the original code). Each view
    (diag / triu / rowmean) of a classical encoding is scaled independently;
    a quantum twobody tensor is scaled per timestep over all its entries
    (zero diagonal included) and its views are taken afterwards.
    """
    out = [copy.copy(d) for d in dataset]
    for g in groups:
        if g in QUANTUM_ATTRS:
            out = normalize_correlations(out, *QUANTUM_ATTRS[g], fit_idx=fit_idx)
        elif g in CLASSICAL_GROUPS:
            for v in VIEWS:
                key = f"{g}_{v}"
                arrays = [_to_numpy(getattr(d, key)) for d in out]
                for d, arr in zip(out, _normalize_per_step(arrays, fit_idx)):
                    setattr(d, key, torch.tensor(arr, dtype=torch.float32))
    summaries = [g.removesuffix("_summary") for g in groups if g in SUMMARY_GROUPS]
    if summaries:
        compute_and_normalize_summary_stats(out, summaries, fit_idx)
    return out


# ---------------------------------------------------------------------------
# PE builders (unchanged from the original)
# ---------------------------------------------------------------------------


def build_node_pe(data: Data, pe_type: str) -> np.ndarray:
    """Build node-level PE (N, D) for a single graph."""
    # Controls (no access to onebody_features)
    if pe_type == "Interaction":
        return _to_numpy(data.interaction_diag)
    elif pe_type == "Distance":
        return _to_numpy(data.distance_diag)

    ob = _to_numpy(data.onebody_features)
    N = ob.shape[0]
    T = ob.shape[1]

    if pe_type == "random":
        return np.random.rand(N, T)
    elif pe_type == "FM":
        return ob
    elif pe_type == "AFM":
        return _to_numpy(data.onebody_features_afm)
    elif pe_type == "HW_AFM":
        return _to_numpy(data.onebody_features_hw_afm)
    elif pe_type == "RRWP_HW_AFM":
        return np.concatenate(
            [_to_numpy(data.rrwp_diag), _to_numpy(data.onebody_features_hw_afm)],
            axis=1,
        )
    elif pe_type == "FM_AFM":
        return np.concatenate([ob, _to_numpy(data.onebody_features_afm)], axis=1)
    elif pe_type == "RRWP":
        return _to_numpy(data.rrwp_diag)
    elif pe_type == "RRWP_FM":
        return np.concatenate([_to_numpy(data.rrwp_diag), ob], axis=1)
    elif pe_type == "RRWP_AFM":
        return np.concatenate(
            [_to_numpy(data.rrwp_diag), _to_numpy(data.onebody_features_afm)],
            axis=1,
        )
    elif pe_type == "RRWP_FM_AFM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                ob,
                _to_numpy(data.onebody_features_afm),
            ],
            axis=1,
        )
    elif pe_type == "Laplacian":
        return _to_numpy(data.laplacian_diag)
    elif pe_type == "RRWP_Laplacian":
        return np.concatenate([_to_numpy(data.rrwp_diag), _to_numpy(data.laplacian_diag)], axis=1)
    elif pe_type == "FM_mean":
        return _to_numpy(data.fm_mean_diag)
    elif pe_type == "AFM_mean":
        return _to_numpy(data.afm_mean_diag)
    elif pe_type == "FM_stat":
        return _to_numpy(data.fm_stat_diag)
    elif pe_type == "AFM_stat":
        return _to_numpy(data.afm_stat_diag)
    elif pe_type == "Ring":
        return _to_numpy(data.ring_diag)
    elif pe_type == "RRWP_Laplacian_Ring":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.ring_diag),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.ring_diag),
                ob,
                _to_numpy(data.onebody_features_afm),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.ring_diag),
                _to_numpy(data.interaction_diag),
                _to_numpy(data.distance_diag),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.ring_diag),
                _to_numpy(data.interaction_diag),
                _to_numpy(data.distance_diag),
                ob,
                _to_numpy(data.onebody_features_afm),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres":
        return np.concatenate(
            [
                build_node_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_fmres),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_AFMres":
        return np.concatenate(
            [
                build_node_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_afmres),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres_AFMres":
        return np.concatenate(
            [
                build_node_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_fmres),
                _to_numpy(data.onebody_features_afmres),
            ],
            axis=1,
        )

    raise ValueError(f"Unknown pe_type: {pe_type}")


def build_edge_pe(data: Data, pe_type: str) -> np.ndarray:
    """Build edge-level PE (M, D) from upper triangular of twobody features."""
    # Controls (no access to twobody_features)
    if pe_type == "Interaction":
        return _to_numpy(data.interaction_triu)
    elif pe_type == "Distance":
        return _to_numpy(data.distance_triu)

    tb = _to_numpy(data.twobody_features)
    N = tb.shape[0]
    T = tb.shape[2]

    if pe_type == "random":
        M = N * (N - 1) // 2
        return np.random.rand(M, T)
    elif pe_type == "FM":
        return _extract_triu(tb)
    elif pe_type == "AFM":
        return _extract_triu(_to_numpy(data.twobody_features_afm))
    elif pe_type == "HW_AFM":
        return _extract_triu(_to_numpy(data.twobody_features_hw_afm))
    elif pe_type == "RRWP_HW_AFM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _extract_triu(_to_numpy(data.twobody_features_hw_afm)),
            ],
            axis=1,
        )
    elif pe_type == "FM_AFM":
        return np.concatenate(
            [
                _extract_triu(tb),
                _extract_triu(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP":
        return _to_numpy(data.rrwp_triu)
    elif pe_type == "RRWP_FM":
        return np.concatenate([_to_numpy(data.rrwp_triu), _extract_triu(tb)], axis=1)
    elif pe_type == "RRWP_AFM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _extract_triu(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_FM_AFM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _extract_triu(tb),
                _extract_triu(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "Laplacian":
        return _to_numpy(data.laplacian_triu)
    elif pe_type == "RRWP_Laplacian":
        return np.concatenate([_to_numpy(data.rrwp_triu), _to_numpy(data.laplacian_triu)], axis=1)
    elif pe_type == "FM_mean":
        return _to_numpy(data.fm_mean_triu)
    elif pe_type == "AFM_mean":
        return _to_numpy(data.afm_mean_triu)
    elif pe_type == "FM_stat":
        return _to_numpy(data.fm_stat_triu)
    elif pe_type == "AFM_stat":
        return _to_numpy(data.afm_stat_triu)
    elif pe_type == "Ring":
        return _to_numpy(data.ring_triu)
    elif pe_type == "RRWP_Laplacian_Ring":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _to_numpy(data.laplacian_triu),
                _to_numpy(data.ring_triu),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _to_numpy(data.laplacian_triu),
                _to_numpy(data.ring_triu),
                _extract_triu(tb),
                _extract_triu(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _to_numpy(data.laplacian_triu),
                _to_numpy(data.ring_triu),
                _to_numpy(data.interaction_triu),
                _to_numpy(data.distance_triu),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_triu),
                _to_numpy(data.laplacian_triu),
                _to_numpy(data.ring_triu),
                _to_numpy(data.interaction_triu),
                _to_numpy(data.distance_triu),
                _extract_triu(tb),
                _extract_triu(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres":
        return np.concatenate(
            [
                build_edge_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _extract_triu(_to_numpy(data.twobody_features_fmres)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_AFMres":
        return np.concatenate(
            [
                build_edge_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _extract_triu(_to_numpy(data.twobody_features_afmres)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres_AFMres":
        return np.concatenate(
            [
                build_edge_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _extract_triu(_to_numpy(data.twobody_features_fmres)),
                _extract_triu(_to_numpy(data.twobody_features_afmres)),
            ],
            axis=1,
        )

    raise ValueError(f"Unknown pe_type: {pe_type}")


def _row_mean(twobody: np.ndarray) -> np.ndarray:
    return twobody.mean(axis=1)


def build_graph_pe(data: Data, pe_type: str) -> np.ndarray:
    """Build graph-level PE as per-node features (N, D).

    Concatenates node PE + row-mean of twobody PE per node.
    """
    # Controls (no access to onebody_features)
    if pe_type == "Interaction":
        return np.concatenate(
            [
                _to_numpy(data.interaction_diag),
                _to_numpy(data.interaction_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "Distance":
        return np.concatenate(
            [
                _to_numpy(data.distance_diag),
                _to_numpy(data.distance_rowmean),
            ],
            axis=1,
        )

    ob = _to_numpy(data.onebody_features)
    N = ob.shape[0]
    T = ob.shape[1]

    if pe_type == "random":
        return np.random.rand(N, 2 * T)
    elif pe_type == "FM":
        return np.concatenate([ob, _row_mean(_to_numpy(data.twobody_features))], axis=1)
    elif pe_type == "AFM":
        afm_ob = _to_numpy(data.onebody_features_afm)
        return np.concatenate([afm_ob, _row_mean(_to_numpy(data.twobody_features_afm))], axis=1)
    elif pe_type == "HW_AFM":
        hw_ob = _to_numpy(data.onebody_features_hw_afm)
        return np.concatenate([hw_ob, _row_mean(_to_numpy(data.twobody_features_hw_afm))], axis=1)
    elif pe_type == "RRWP_HW_AFM":
        hw_ob = _to_numpy(data.onebody_features_hw_afm)
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                hw_ob,
                _row_mean(_to_numpy(data.twobody_features_hw_afm)),
            ],
            axis=1,
        )
    elif pe_type == "FM_AFM":
        return np.concatenate(
            [
                ob,
                _to_numpy(data.onebody_features_afm),
                _row_mean(_to_numpy(data.twobody_features)),
                _row_mean(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_FM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                ob,
                _row_mean(_to_numpy(data.twobody_features)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_AFM":
        afm_ob = _to_numpy(data.onebody_features_afm)
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                afm_ob,
                _row_mean(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_FM_AFM":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                ob,
                _to_numpy(data.onebody_features_afm),
                _row_mean(_to_numpy(data.twobody_features)),
                _row_mean(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "Laplacian":
        return np.concatenate(
            [
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "FM_mean":
        return np.concatenate(
            [
                _to_numpy(data.fm_mean_diag),
                _to_numpy(data.fm_mean_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "AFM_mean":
        return np.concatenate(
            [
                _to_numpy(data.afm_mean_diag),
                _to_numpy(data.afm_mean_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "FM_stat":
        return np.concatenate(
            [
                _to_numpy(data.fm_stat_diag),
                _to_numpy(data.fm_stat_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "AFM_stat":
        return np.concatenate(
            [
                _to_numpy(data.afm_stat_diag),
                _to_numpy(data.afm_stat_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "Ring":
        return np.concatenate(
            [
                _to_numpy(data.ring_diag),
                _to_numpy(data.ring_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
                _to_numpy(data.ring_diag),
                _to_numpy(data.ring_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
                _to_numpy(data.ring_diag),
                _to_numpy(data.ring_rowmean),
                ob,
                _to_numpy(data.onebody_features_afm),
                _row_mean(_to_numpy(data.twobody_features)),
                _row_mean(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
                _to_numpy(data.ring_diag),
                _to_numpy(data.ring_rowmean),
                _to_numpy(data.interaction_diag),
                _to_numpy(data.interaction_rowmean),
                _to_numpy(data.distance_diag),
                _to_numpy(data.distance_rowmean),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_Quantum":
        return np.concatenate(
            [
                _to_numpy(data.rrwp_diag),
                _to_numpy(data.rrwp_rowmean),
                _to_numpy(data.laplacian_diag),
                _to_numpy(data.laplacian_rowmean),
                _to_numpy(data.ring_diag),
                _to_numpy(data.ring_rowmean),
                _to_numpy(data.interaction_diag),
                _to_numpy(data.interaction_rowmean),
                _to_numpy(data.distance_diag),
                _to_numpy(data.distance_rowmean),
                ob,
                _to_numpy(data.onebody_features_afm),
                _row_mean(_to_numpy(data.twobody_features)),
                _row_mean(_to_numpy(data.twobody_features_afm)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres":
        return np.concatenate(
            [
                build_graph_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_fmres),
                _row_mean(_to_numpy(data.twobody_features_fmres)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_AFMres":
        return np.concatenate(
            [
                build_graph_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_afmres),
                _row_mean(_to_numpy(data.twobody_features_afmres)),
            ],
            axis=1,
        )
    elif pe_type == "RRWP_Laplacian_Ring_Geometric_FMres_AFMres":
        return np.concatenate(
            [
                build_graph_pe(data, "RRWP_Laplacian_Ring_Geometric"),
                _to_numpy(data.onebody_features_fmres),
                _to_numpy(data.onebody_features_afmres),
                _row_mean(_to_numpy(data.twobody_features_fmres)),
                _row_mean(_to_numpy(data.twobody_features_afmres)),
            ],
            axis=1,
        )

    raise ValueError(f"Unknown pe_type: {pe_type}")
