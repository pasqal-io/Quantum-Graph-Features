"""Descripteurs beyond-WL : motifs de treewidth ≥ 3 et orbites d'automorphismes."""

from __future__ import annotations

import networkx as nx
import numpy as np

from qgf.reconstruction.data import pair_index
from qgf.reconstruction.features_beyond import motif_features, symmetry_features


def _pairs(num_nodes):
    return pair_index(num_nodes)


def test_motif_k4_fills_co_clique_on_every_edge():
    """On K_4, every pair is an edge and sees its 2 other nodes adjacent → 1
    K_4."""
    adjacency = np.array(nx.to_numpy_array(nx.complete_graph(4)))
    matrix, labels = motif_features(adjacency, _pairs(4))
    assert labels == ("co_clique4", "co_open4")
    # 6 pairs, all edges: co_clique4 = 1 (one edge inside the common neighbourhood).
    assert np.allclose(matrix[:, 0], 1.0)
    assert np.allclose(matrix[:, 1], 0.0)


def test_motif_pure_c4_has_no_second_order_structure():
    """A chordless 4-cycle: no triangle in a common neighbourhood."""
    adjacency = np.array(nx.to_numpy_array(nx.cycle_graph(4)))
    matrix, _ = motif_features(adjacency, _pairs(4))
    assert np.allclose(matrix, 0.0)


def test_motif_diamond_fills_co_open_on_the_diagonal():
    """C_4 + chord 1–3: the non-adjacent pair (0,2) has adjacent {1,3} → co_open4 =
    1."""
    graph = nx.cycle_graph(4)
    graph.add_edge(1, 3)
    adjacency = np.array(nx.to_numpy_array(graph))
    rows, cols = _pairs(4)
    matrix, _ = motif_features(adjacency, (rows, cols))
    pair_02 = np.flatnonzero((rows == 0) & (cols == 2))[0]
    assert matrix[pair_02, 1] == 1.0  # co_open4
    assert matrix[pair_02, 0] == 0.0  # (0,2) is not an edge


def test_symmetry_cycle_puts_all_nodes_in_one_orbit():
    graph = nx.cycle_graph(4)
    matrix, labels = symmetry_features(graph, _pairs(4))
    assert labels == ("same_orbit", "orbit_size_min", "orbit_size_max")
    assert np.allclose(matrix[:, 0], 1.0)  # tous co-orbitaux
    assert np.allclose(matrix[:, 1], 4.0)
    assert np.allclose(matrix[:, 2], 4.0)


def test_symmetry_path_separates_center_from_endpoints():
    """P_3 (0–1–2) : orbites {0,2} et {1}.

    (0,2) co-orbitaux, (0,1) non.
    """
    graph = nx.path_graph(3)
    rows, cols = _pairs(3)
    matrix, _ = symmetry_features(graph, (rows, cols))
    pair_02 = np.flatnonzero((rows == 0) & (cols == 2))[0]
    pair_01 = np.flatnonzero((rows == 0) & (cols == 1))[0]
    assert matrix[pair_02, 0] == 1.0
    assert matrix[pair_01, 0] == 0.0


def test_asymmetric_graph_has_trivial_orbits():
    """A graph with no non-trivial automorphism: every node alone in its orbit."""
    # Spider with legs of distinct lengths 1, 2, 3 from centre 0: no symmetry
    # possible, since the three branches are distinguishable by length.
    graph = nx.Graph([(0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (5, 6)])
    matrix, _ = symmetry_features(graph, _pairs(7))
    assert np.allclose(matrix[:, 0], 0.0)
    assert np.allclose(matrix[:, 1], 1.0)
    assert np.allclose(matrix[:, 2], 1.0)
