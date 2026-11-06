"""Ring basis.

MUTAG and ZINC molecules are dominated by rings, and the quantum correlation along a
ring interferes qualitatively differently from a chain: typically the kind of structure
that a sum of powers of A captures poorly at moderate K.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import ORDER_ALWAYS, DesignBlock, register_basis

MIN_RING_SIZE: int = 3


def _cycle_distance_matrix(graph: nx.Graph, cycle: list[int]) -> dict[int, dict[int, int]]:
    """Distances along the cycle, computed in the induced subgraph.

    For a chordless cycle this is the cyclic distance min(l, L - l); the induced-subgraph
    definition stays well posed with chords.
    """
    induced = graph.subgraph(cycle)
    return dict(nx.all_pairs_shortest_path_length(induced))


def _is_aromatic(
    graph: nx.Graph,
    cycle: list[int],
    bond_types: np.ndarray | None,
    aromatic_index: int | None,
) -> bool:
    """Do all bonds of the induced subgraph carry the aromatic type?"""
    if bond_types is None or aromatic_index is None:
        return False
    induced = graph.subgraph(cycle)
    edges = list(induced.edges())
    if not edges:
        return False
    return all(int(bond_types[left, right]) == aromatic_index for left, right in edges)


def ring_features(
    graph: nx.Graph,
    pair_index: tuple[np.ndarray, np.ndarray],
    max_ring_size: int = 8,
    bond_types: np.ndarray | None = None,
    aromatic_index: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Returns (n_pairs, F_ring) for the given pairs (i<j), and the column names.

    Columns, per ring size L = 3..max_ring_size:
      - in_ring_L[i, j]  : 1 if i and j belong to the SAME ring of size L
      - n_rings_L[i, j]  : number of such shared rings
      - ring_dist_L[i,j] : distance along the ring (0 if not co-cyclic)
      - cos_L, sin_L     : normalised angular distance cos(2 pi l/L), sin(2 pi l/L), the
                           natural form of the eigenmodes of a ring
      - closed_L_i, closed_L_j : closed walks (A^L)_ii and (A^L)_jj at the endpoints
    Aggregated columns:
      - ring_member_i, ring_member_j : ring membership of each endpoint
      - fused        : the two atoms are in distinct fused rings
      - aromatic     : the shared ring is aromatic (if encoded by the dataset)
    """
    rows, cols = pair_index
    num_pairs = rows.size
    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)
    sizes = tuple(range(MIN_RING_SIZE, max_ring_size + 1))

    cycles = [sorted(cycle) for cycle in nx.minimum_cycle_basis(graph)]
    by_size: dict[int, list[list[int]]] = {size: [] for size in sizes}
    for cycle in cycles:
        if len(cycle) in by_size:
            by_size[len(cycle)].append(cycle)

    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    closed_walks: dict[int, np.ndarray] = {}
    power = np.eye(num_nodes)
    for order in range(1, max_ring_size + 1):
        power = power @ adjacency
        closed_walks[order] = np.diag(power).copy()

    columns: list[np.ndarray] = []
    labels: list[str] = []
    member = np.zeros(num_nodes)
    aromatic = np.zeros(num_pairs)

    for size in sizes:
        in_ring = np.zeros(num_pairs)
        counts = np.zeros(num_pairs)
        distance = np.zeros(num_pairs)
        for cycle in by_size[size]:
            mask = np.zeros(num_nodes, dtype=bool)
            mask[cycle] = True
            member[cycle] = 1.0
            shared = mask[rows] & mask[cols]
            if not np.any(shared):
                continue
            lengths = _cycle_distance_matrix(graph, cycle)
            pair_distance = np.zeros(num_pairs)
            selected = np.flatnonzero(shared)
            pair_distance[selected] = [lengths[int(rows[p])][int(cols[p])] for p in selected]
            # Several rings of the same size can share the pair: keep the minimal distance
            # and count the rings separately.
            previous = distance[selected]
            distance[selected] = np.where(
                previous > 0,
                np.minimum(previous, pair_distance[selected]),
                pair_distance[selected],
            )
            in_ring[shared] = 1.0
            counts[shared] += 1.0
            if _is_aromatic(graph, cycle, bond_types, aromatic_index):
                aromatic[shared] = 1.0

        angle = 2.0 * np.pi * distance / size
        columns.extend(
            [
                in_ring,
                counts,
                distance,
                np.where(in_ring > 0, np.cos(angle), 0.0),
                np.where(in_ring > 0, np.sin(angle), 0.0),
                closed_walks[size][rows],
                closed_walks[size][cols],
            ],
        )
        labels.extend(
            [
                f"in_ring_{size}",
                f"n_rings_{size}",
                f"ring_dist_{size}",
                f"ring_cos_{size}",
                f"ring_sin_{size}",
                f"closed_{size}_i",
                f"closed_{size}_j",
            ],
        )

    fused = np.zeros(num_pairs, dtype=bool)
    for left in range(len(cycles)):
        for right in range(left + 1, len(cycles)):
            first, second = set(cycles[left]), set(cycles[right])
            if len(first & second) < 2:
                continue
            mask_first = np.zeros(num_nodes, dtype=bool)
            mask_first[list(first)] = True
            mask_second = np.zeros(num_nodes, dtype=bool)
            mask_second[list(second)] = True
            fused |= (mask_first[rows] & mask_second[cols]) | (mask_second[rows] & mask_first[cols])

    columns.extend([member[rows], member[cols], fused.astype(float), aromatic])
    labels.extend(["ring_member_i", "ring_member_j", "fused", "aromatic"])
    return np.stack(columns, axis=1), tuple(labels)


@register_basis("ring", phase="P2")
def ring_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Ring basis: co-membership, cyclic distance, ring modes, fusion, aromaticity."""
    # The aromatic column stays zero when the dataset does not encode aromaticity.
    matrix, labels = ring_features(
        sample.graph,
        sample.pair_index,
        max_ring_size=context.max_ring_size,
        bond_types=sample.bond_types,
        aromatic_index=sample.aromatic_bond_index,
    )
    return DesignBlock(
        matrix=matrix,
        labels=labels,
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )
