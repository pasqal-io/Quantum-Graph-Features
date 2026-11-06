"""Beyond-WL descriptors used to characterise the reconstruction residual.

The classical ceiling is, mathematically, the set of matrix functions of A and L evaluated
on the pair (i, j), plus short cycles and metric distances: roughly 2-WL / spectral. This
module provides invariants that **separate** that level:

- ``motif``: number of edges inside the common neighbourhood of (i, j). A neighbourhood-
  restricted triangle count, i.e. a treewidth >= 3 motif (K_4 / diamond); it is not a
  function of the sequence {(A^k)_ij}.
- ``symmetry``: orbits of the automorphism group. Two nodes of the same orbit are
  indistinguishable by any invariant quantity.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import ORDER_ALWAYS, DesignBlock, register_basis

#: Above this size automorphisms are not searched (VF2 cost); orbits are then trivial.
AUTOMORPHISM_NODE_CAP: int = 60
#: Guard on the enumeration of automorphisms of a very symmetric graph.
MAX_AUTOMORPHISMS: int = 50_000


def motif_features(
    adjacency: np.ndarray,
    pair_index: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """(n_pairs, 2): edges inside the common neighbourhood of (i, j), split by A_ij.

    For the pair (i, j), let S = N(i) & N(j) minus {i, j} and e = number of edges inside S:

      - ``co_clique4`` = e if (i, j) is an edge (each edge of S closes a K_4 on ij);
      - ``co_open4``   = e otherwise (each edge of S closes a diamond of diagonal ij).
    """
    boolean = np.asarray(adjacency) != 0
    rows, cols = pair_index
    num_pairs = rows.size
    co_clique4 = np.zeros(num_pairs)
    co_open4 = np.zeros(num_pairs)
    for pair in range(num_pairs):
        i, j = int(rows[pair]), int(cols[pair])
        common = boolean[i] & boolean[j]
        common[i] = common[j] = False
        idx = np.flatnonzero(common)
        if idx.size < 2:
            continue
        internal_edges = boolean[np.ix_(idx, idx)].sum() / 2.0
        if boolean[i, j]:
            co_clique4[pair] = internal_edges
        else:
            co_open4[pair] = internal_edges
    return np.stack([co_clique4, co_open4], axis=1), ("co_clique4", "co_open4")


def _automorphism_orbits(graph: nx.Graph) -> dict[int, int]:
    """Automorphism-orbit representative of every node (union-find over the images).

    Above :data:`AUTOMORPHISM_NODE_CAP`, or past :data:`MAX_AUTOMORPHISMS`, orbits stay
    partial: a conservative bound that never invents symmetry.
    """
    parent = {node: node for node in graph.nodes()}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    if graph.number_of_nodes() <= AUTOMORPHISM_NODE_CAP:
        for count, mapping in enumerate(GraphMatcher(graph, graph).isomorphisms_iter()):
            for source, target in mapping.items():
                union(source, target)
            if count + 1 >= MAX_AUTOMORPHISMS:
                break
    return {node: find(node) for node in graph.nodes()}


def symmetry_features(
    graph: nx.Graph,
    pair_index: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """(n_pairs, 3): orbit co-membership and orbit sizes (symmetric in i, j)."""
    orbit = _automorphism_orbits(graph)
    sizes = Counter(orbit.values())
    node_size = {node: sizes[orbit[node]] for node in graph.nodes()}

    rows, cols = pair_index
    same = np.array(
        [1.0 if orbit[int(i)] == orbit[int(j)] else 0.0 for i, j in zip(rows, cols)],
    )
    size_i = np.array([node_size[int(i)] for i in rows], dtype=float)
    size_j = np.array([node_size[int(j)] for j in cols], dtype=float)
    matrix = np.stack(
        [same, np.minimum(size_i, size_j), np.maximum(size_i, size_j)],
        axis=1,
    )
    return matrix, ("same_orbit", "orbit_size_min", "orbit_size_max")


@register_basis("motif", phase="P4")
def motif_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Beyond-WL motif: edges of the common neighbourhood (K_4 / diamond on the pair)."""
    matrix, labels = motif_features(sample.adjacency, sample.pair_index)
    return DesignBlock(
        matrix=matrix,
        labels=labels,
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


@register_basis("symmetry", phase="P4")
def symmetry_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Automorphism orbits: co-membership and orbit sizes."""
    matrix, labels = symmetry_features(sample.graph, sample.pair_index)
    return DesignBlock(
        matrix=matrix,
        labels=labels,
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )
