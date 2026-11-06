import networkx as nx
import numpy as np
import pytest

from qgf.intrinsic import metrics as m
from qgf.intrinsic.pe_construction import data_to_graph


def test_registry_has_the_15_paper_tasks():
    assert len(m.TASKS) == 15
    assert m.get_category("degree_prediction") == "node_prediction"
    assert m.get_category("link_prediction") == "edge_prediction"
    assert m.get_category("graph_diameter") == "graph_prediction"
    with pytest.raises(ValueError, match="Unknown benchmark"):
        m.get_category("logp")


def test_node_targets(small_graph, medium_graph):
    tri, tree = data_to_graph(small_graph), data_to_graph(medium_graph)
    np.testing.assert_array_equal(m.degree_targets(tri), [2, 2, 2])
    assert m.degree_targets(tree)[1] == 3
    np.testing.assert_array_equal(m.bridge_node_targets(tri), [False] * 3)
    assert m.bridge_node_targets(tree).sum() > 0
    np.testing.assert_array_equal(m.eccentricity_targets(tri), [1, 1, 1])
    np.testing.assert_array_equal(m.smallest_cycle_node_targets(tri), [3.0] * 3)
    np.testing.assert_array_equal(m.smallest_cycle_node_targets(tree), [0.0] * 5)


def test_edge_targets(small_graph, medium_graph):
    tri, tree = data_to_graph(small_graph), data_to_graph(medium_graph)
    np.testing.assert_array_equal(m.link_prediction_targets(tri, 3), [1, 1, 1])
    lp = m.link_prediction_targets(tree, 5)
    assert lp.shape == (10,) and lp.sum() == 4
    np.testing.assert_array_equal(m.smallest_cycle_edge_targets(tri, 3), [3.0] * 3)
    assert m.smallest_cycle_edge_targets(tree, 5).sum() == 0
    assert m.bridge_edge_targets(tree, 5).sum() == 4
    np.testing.assert_array_equal(m.local_edge_connectivity_targets(tri, 3), [2, 2, 2])


def test_graph_targets(small_graph, medium_graph):
    tri, tree = data_to_graph(small_graph), data_to_graph(medium_graph)
    assert m.num_cycles_target(tri) == 1.0 and m.num_cycles_target(tree) == 0.0
    assert m.graph_diameter_target(tri) == 1.0 and m.graph_diameter_target(tree) == 3.0
    np.testing.assert_allclose(m.graph_density_target(tri), 1.0)
    np.testing.assert_allclose(m.wiener_index_target(tri), 1.0)
    assert m.algebraic_connectivity_target(tri) > 0.0


def test_compute_targets_shapes():
    graphs = [nx.cycle_graph(4), nx.path_graph(3)]
    assert [t.shape for t in m.compute_targets("degree_prediction", graphs)] == [(4,), (3,)]
    assert [t.shape for t in m.compute_targets("link_prediction", graphs)] == [(6,), (3,)]
    assert m.compute_targets("num_cycles", graphs).tolist() == [1.0, 0.0]
