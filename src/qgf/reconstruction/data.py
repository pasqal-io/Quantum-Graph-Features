"""Input contract of the experiment and loading of the released datasets.

A :class:`GraphSample` exposes exactly the fields the bases need: ``adjacency`` (N, N),
``correlations`` (N_t, N, N), ``times_ns`` (N_t,), ``embedded_position`` (N, 2) and
``graph_id``. Chemical attributes (atom types, bond types) are carried too, for the
chemical control basis and the aromatic flag of the ring basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import networkx as nx
import numpy as np

from qgf.reconstruction.config import ZZ_TO_OCCUPATION_SCALE

#: Index of the aromatic bond type in the one-hot ``edge_attr``, per dataset. ``None`` when
#: the dataset does not encode aromaticity (ZINC only has single/double/triple).
AROMATIC_BOND_INDEX: dict[str, int | None] = {
    "MUTAG": 0,
    "ZINC_small": None,
}

#: Released dataset name per (dataset, quench). ZINC is the union train + val + test, in
#: that order: reconstruction is not a learning task, the splits are pooled.
_RELEASED_NAMES: dict[str, str] = {
    "MUTAG": "mutag_{quench}",
    "ZINC_small": "zinc_{quench}",
}
_ZINC_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(eq=False)
class GraphSample:
    """A graph and its tensor of quantum correlations.

    Attributes:
        graph_id: index in the (pooled) dataset, for traceability.
        adjacency: (N, N) unweighted adjacency, symmetric, zero diagonal.
        correlations: (N_t, N, N) tensor C_ij(t_nu), symmetric in (i, j).
        times_ns: (N_t,) time grid in ns.
        embedded_position: (N, 2) atom layout in um, or None.
        node_types: (N,) atom-type index, or None.
        bond_types: (N, N) bond-type index, -1 off bonds, or None.
        dataset: source dataset name, which fixes the bond-type encoding.
        occupations: (N, N_t) one-body features (stored ``onebody_features``, node-major),
            or None. Target of the node-level residual.
    """

    graph_id: int
    adjacency: np.ndarray
    correlations: np.ndarray
    times_ns: np.ndarray
    embedded_position: np.ndarray | None = None
    node_types: np.ndarray | None = None
    bond_types: np.ndarray | None = None
    dataset: str = ""
    occupations: np.ndarray | None = None

    @property
    def aromatic_bond_index(self) -> int | None:
        """One-hot index of the aromatic bond type, or None if not encoded."""
        return AROMATIC_BOND_INDEX.get(self.dataset)

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def num_pairs(self) -> int:
        """Number of independent off-diagonal entries, N(N-1)/2."""
        n = self.num_nodes
        return n * (n - 1) // 2

    @cached_property
    def pair_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices (i < j) of the strict upper triangle, lexicographic order."""
        return np.triu_indices(self.num_nodes, k=1)

    @cached_property
    def graph(self) -> nx.Graph:
        """networkx graph on nodes 0..N-1 (edges of the adjacency)."""
        return nx.from_numpy_array(np.asarray(self.adjacency))

    def targets(self) -> np.ndarray:
        """Target y_alpha: (n_pairs, N_t), strict upper triangle per time step."""
        rows, cols = self.pair_index
        return np.ascontiguousarray(self.correlations[:, rows, cols].T)


def pair_index(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i<j) of the strict upper triangle, lexicographic order."""
    return np.triu_indices(num_nodes, k=1)


def _adjacency_from_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Unweighted symmetric adjacency with zero diagonal, from an edge_index."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    edges = np.asarray(edge_index)
    adjacency[edges[0], edges[1]] = 1
    adjacency[edges[1], edges[0]] = 1
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _bond_types_from_edge_attr(
    edge_index: np.ndarray,
    edge_attr: np.ndarray | None,
    num_nodes: int,
) -> np.ndarray | None:
    """(N, N) matrix of bond-type indices, -1 off bonds."""
    if edge_attr is None:
        return None
    attr = np.asarray(edge_attr)
    types = attr.argmax(axis=1) if attr.ndim == 2 else attr.astype(int)
    matrix = np.full((num_nodes, num_nodes), -1, dtype=np.int16)
    edges = np.asarray(edge_index)
    matrix[edges[0], edges[1]] = types
    matrix[edges[1], edges[0]] = types
    np.fill_diagonal(matrix, -1)
    return matrix


def _node_types(data) -> np.ndarray | None:
    """Atom-type index from the one-hot ``x``."""
    raw = getattr(data, "x", None)
    if raw is None:
        return None
    features = np.asarray(raw)
    if features.ndim == 2 and features.shape[1] > 1:
        return features.argmax(axis=1).astype(np.int16)
    return features.reshape(-1).astype(np.int16)


def sample_from_data(
    data,
    graph_id: int,
    times_ns: np.ndarray,
    target_scale: float = ZZ_TO_OCCUPATION_SCALE,
    dataset: str = "",
) -> GraphSample:
    """Converts a released ``Data`` object (time-major features) to a GraphSample."""
    num_nodes = int(np.asarray(data.x).shape[0])
    correlations = np.asarray(data.twobody_features, dtype=float) * float(target_scale)
    if correlations.shape != (len(times_ns), num_nodes, num_nodes):
        raise ValueError(
            f"Graph {graph_id}: twobody_features of shape {correlations.shape}, expected "
            f"({len(times_ns)}, {num_nodes}, {num_nodes})",
        )
    onebody = getattr(data, "onebody_features", None)
    occupations = None
    if onebody is not None:
        occupations = np.ascontiguousarray(np.asarray(onebody, dtype=float).T)
    edge_index = np.asarray(data.edge_index)
    pos = getattr(data, "pos", None)
    return GraphSample(
        graph_id=graph_id,
        adjacency=_adjacency_from_edge_index(edge_index, num_nodes),
        correlations=correlations,
        times_ns=np.asarray(times_ns, dtype=float),
        embedded_position=None if pos is None else np.asarray(pos, dtype=float),
        node_types=_node_types(data),
        bond_types=_bond_types_from_edge_attr(
            edge_index,
            getattr(data, "edge_attr", None),
            num_nodes,
        ),
        dataset=dataset,
        occupations=occupations,
    )


def load_raw_dataset(
    dataset: str,
    quench: str,
    suffix: str = "",
    root: str | Path | None = None,
) -> tuple[list, dict]:
    """Released ``Data`` objects and meta for a dataset and a quench.

    ``suffix="_residual"`` loads the released residual datasets. ZINC splits are
    concatenated in the order train, val, test, and ``meta["split_sizes"]`` records them.
    """
    from qgf.data import DEFAULT_ROOT, load

    if quench not in ("AFM", "FM"):
        raise ValueError(f"Unknown quench: {quench}")
    if dataset not in _RELEASED_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of {sorted(_RELEASED_NAMES)}")
    name = _RELEASED_NAMES[dataset].format(quench=quench.lower()) + suffix
    root = DEFAULT_ROOT if root is None else Path(root)
    if dataset != "ZINC_small":
        return load(name, root=root)
    graphs: list = []
    sizes = []
    meta: dict = {}
    for split in _ZINC_SPLITS:
        part, meta = load(name, split, root=root)
        graphs.extend(part)
        sizes.append(len(part))
    return graphs, {**meta, "split": None, "split_sizes": sizes}


def load_graph_samples(
    dataset: str,
    quench: str,
    times_ns: np.ndarray | None = None,
    target_scale: float = ZZ_TO_OCCUPATION_SCALE,
    max_graphs: int | None = None,
    root: str | Path | None = None,
) -> list[GraphSample]:
    """Loads a released dataset and converts it to GraphSamples.

    ``times_ns=None`` uses the true grid of ``meta["times_ns"]``. An explicit grid only
    relabels the time axis: it must have the same length.
    """
    raw, meta = load_raw_dataset(dataset, quench, root=root)
    if times_ns is None:
        times_ns = np.asarray(meta["times_ns"], dtype=float)
    if max_graphs is not None:
        raw = raw[:max_graphs]
    return [
        sample_from_data(data, graph_id, times_ns, target_scale, dataset=dataset)
        for graph_id, data in enumerate(raw)
    ]


@dataclass(frozen=True)
class DatasetDiagnostics:
    """Input checks that condition the interpretation of the results."""

    dataset: str
    quench: str
    num_graphs: int
    num_pairs: int
    num_timesteps: int
    min_nodes: int
    max_nodes: int
    target_min: float
    target_max: float
    times_first_ns: float
    times_last_ns: float

    def summary(self) -> str:
        return (
            f"{self.dataset}/{self.quench}: {self.num_graphs} graphs, P={self.num_pairs}, "
            f"N_t={self.num_timesteps}, N in [{self.min_nodes}, {self.max_nodes}], "
            f"C in [{self.target_min:+.4f}, {self.target_max:+.4f}], "
            f"t in [{self.times_first_ns:.0f}, {self.times_last_ns:.0f}] ns"
        )


def _slices_symmetric(array: np.ndarray, tol: float = 1e-8) -> bool:
    """True if every slice ``array[k]`` is symmetric."""
    return bool(np.allclose(array, np.swapaxes(array, 1, 2), atol=tol, rtol=0.0))


def validate_samples(
    samples: list[GraphSample],
    dataset: str,
    quench: str,
    expected_pairs: int | None = None,
) -> DatasetDiagnostics:
    """Checks the input contract and returns diagnostics.

    Checks symmetry in (i, j), the zero diagonal of the adjacency, finite targets and, if
    ``expected_pairs`` is given, the pair-count sanity test (P = 30 505 on MUTAG).
    """
    if not samples:
        raise ValueError("Empty dataset")

    num_timesteps = samples[0].correlations.shape[0]
    total_pairs = 0
    target_min, target_max = np.inf, -np.inf
    for sample in samples:
        adjacency = sample.adjacency
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError(f"Graph {sample.graph_id}: non-symmetric adjacency")
        if np.any(np.diag(adjacency) != 0):
            raise ValueError(f"Graph {sample.graph_id}: non-zero adjacency diagonal")
        if sample.correlations.shape[0] != num_timesteps:
            raise ValueError(f"Graph {sample.graph_id}: heterogeneous number of time steps")
        if not _slices_symmetric(sample.correlations, tol=1e-6):
            raise ValueError(f"Graph {sample.graph_id}: correlations not symmetric in (i,j)")
        targets = sample.targets()
        if not np.all(np.isfinite(targets)):
            raise ValueError(f"Graph {sample.graph_id}: non-finite targets")
        total_pairs += sample.num_pairs
        target_min = min(target_min, float(targets.min()))
        target_max = max(target_max, float(targets.max()))

    if expected_pairs is not None and total_pairs != expected_pairs:
        raise ValueError(
            f"Pair-count sanity test failed: P={total_pairs} instead of {expected_pairs}",
        )

    nodes = [sample.num_nodes for sample in samples]
    return DatasetDiagnostics(
        dataset=dataset,
        quench=quench,
        num_graphs=len(samples),
        num_pairs=total_pairs,
        num_timesteps=num_timesteps,
        min_nodes=min(nodes),
        max_nodes=max(nodes),
        target_min=target_min,
        target_max=target_max,
        times_first_ns=float(samples[0].times_ns[0]),
        times_last_ns=float(samples[0].times_ns[-1]),
    )
