"""Global descriptors: heat kernel, shortest-path distance, effective resistance, plus a
chemical control basis.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import ORDER_ALWAYS, DesignBlock, register_basis
from qgf.reconstruction.features_base import canonical_eigenvectors, normalized_laplacian


@register_basis("heat", phase="P3")
def heat_kernel_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Heat kernel (exp(-tau L))_ij on a grid of tau (natural spectral resummation)."""
    values, vectors = canonical_eigenvectors(normalized_laplacian(sample.adjacency))
    rows, cols = sample.pair_index
    columns = []
    for tau in context.heat_taus:
        kernel = (vectors * np.exp(-tau * values)[None, :]) @ vectors.T
        columns.append(kernel[rows, cols])
    matrix = np.stack(columns, axis=1)
    return DesignBlock(
        matrix=matrix,
        labels=tuple(f"heat_tau{tau:g}" for tau in context.heat_taus),
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


@register_basis("geodesic", phase="P3")
def geodesic_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Shortest-path distance and its one-hot.

    Disconnected pairs are coded with the sentinel ``max_geodesic + 1``.
    """
    rows, cols = sample.pair_index
    unreachable = context.max_geodesic + 1
    lengths = dict(nx.all_pairs_shortest_path_length(sample.graph))
    distances = np.array(
        [
            min(lengths.get(int(i), {}).get(int(j), unreachable), unreachable)
            for i, j in zip(rows, cols)
        ],
        dtype=float,
    )
    one_hot = np.zeros((rows.size, context.max_geodesic + 1))
    for level in range(1, context.max_geodesic + 1):
        one_hot[:, level - 1] = (distances == level).astype(float)
    one_hot[:, context.max_geodesic] = (distances >= unreachable).astype(float)
    matrix = np.concatenate([distances[:, None], one_hot], axis=1)
    labels = (
        ("geodesic",)
        + tuple(f"geodesic_is_{level}" for level in range(1, context.max_geodesic + 1))
        + ("geodesic_unreachable",)
    )
    return DesignBlock(
        matrix=matrix,
        labels=labels,
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


#: Relative cutoff of the Laplacian pseudo-inverse. The numpy default (1e-15) is fragile:
#: with some LAPACK builds the numerically-zero singular value of a connected graph's
#: Laplacian comes out at ~1e-14 relative and is inverted (values ~1e12). Every nonzero
#: Laplacian singular value of the datasets is >> 1e-10, so this cutoff only removes the
#: null space and reproduces the original values.
PINV_RCOND: float = 1e-10


def laplacian_pinv(laplacian: np.ndarray) -> np.ndarray:
    """Moore-Penrose pseudo-inverse of a graph Laplacian, null space removed robustly."""
    return np.linalg.pinv(laplacian, rcond=PINV_RCOND)


@register_basis("resistance", phase="P3")
def resistance_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Effective resistance Omega_ij = L+_ii + L+_jj - 2 L+_ij (combinatorial Laplacian)."""
    adjacency = np.asarray(sample.adjacency, dtype=float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    pseudo = laplacian_pinv(laplacian)
    rows, cols = sample.pair_index
    diagonal = np.diag(pseudo)
    resistance = diagonal[rows] + diagonal[cols] - 2.0 * pseudo[rows, cols]
    matrix = np.stack([resistance, np.sqrt(np.maximum(resistance, 0.0))], axis=1)
    return DesignBlock(
        matrix=matrix,
        labels=("resistance", "resistance_sqrt"),
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


#: Number of atom types encoded by the chemical basis (upper bound over datasets).
CHEMICAL_NODE_TYPES: int = 28
#: Number of bond types encoded.
CHEMICAL_BOND_TYPES: int = 4


@register_basis("chemical", phase="P3")
def chemical_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Chemical control basis: paired atom types and bond type.

    The QGF pipeline is purely structural and does not see these attributes: if they
    improve the reconstruction, the correlation is dataset-specific, not physical.
    """
    rows, cols = sample.pair_index
    columns: list[np.ndarray] = []
    labels: list[str] = []

    if sample.node_types is not None:
        types = np.asarray(sample.node_types, dtype=int)
        for kind in range(CHEMICAL_NODE_TYPES):
            present_i = (types[rows] == kind).astype(float)
            present_j = (types[cols] == kind).astype(float)
            columns.append(present_i + present_j)
            labels.append(f"atom_count_{kind}")

    if sample.bond_types is not None:
        bonds = np.asarray(sample.bond_types, dtype=int)[rows, cols]
        for kind in range(CHEMICAL_BOND_TYPES):
            columns.append((bonds == kind).astype(float))
            labels.append(f"bond_is_{kind}")

    if not columns:
        raise ValueError(f"Graph {sample.graph_id}: no chemical attribute available")
    matrix = np.stack(columns, axis=1)
    return DesignBlock(
        matrix=matrix,
        labels=tuple(labels),
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )
