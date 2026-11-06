import networkx as nx
import numpy as np
import pytest

from qgf.intrinsic.data_loading import orient
from qgf.intrinsic.pe_construction import (
    C6_PULSER_N70,
    PE_GROUPS,
    RING_SIZES,
    _pairwise_control_tensors,
    build_edge_pe,
    build_graph_pe,
    build_node_pe,
    normalize_dataset,
    ring_membership_tensor,
)

from .conftest import T


def test_orient_always_transposes():
    ob, tb = np.zeros((20, 20)), np.zeros((20, 20, 20))
    ob[3, 5], tb[3, 5, 7] = 1.0, 1.0  # time 3, node 5 (and pair 5-7)
    ob2, tb2 = orient(ob, tb)
    assert ob2[5, 3] == 1.0 and tb2[5, 7, 3] == 1.0


def test_normalize_range_and_per_step(dataset):
    out = normalize_dataset(dataset, ["fm", "rrwp"])
    for t in range(T):
        vals = np.concatenate([d.onebody_features[:, t].numpy() for d in out])
        assert abs(vals.min() + 1) < 1e-5 and abs(vals.max() - 1) < 1e-5
    for d in out:
        assert d.twobody_features.abs().max() <= 1 + 1e-6
        assert d.rrwp_triu.abs().max() <= 1 + 1e-6
    # raw dataset untouched
    assert not np.allclose(dataset[0].onebody_features.numpy(), out[0].onebody_features.numpy())


def test_normalize_fit_idx_uses_only_fit_graphs(dataset):
    fit = [0, 1, 2, 3, 4]
    out = normalize_dataset(dataset, ["fm"], fit_idx=fit)
    for t in range(T):
        fit_vals = np.concatenate([out[i].onebody_features[:, t].numpy() for i in fit])
        assert abs(fit_vals.min() + 1) < 1e-5 and abs(fit_vals.max() - 1) < 1e-5
        raw = [dataset[i].onebody_features[:, t].numpy() for i in fit]
        lo, hi = min(r.min() for r in raw), max(r.max() for r in raw)
        expected = 2 * (dataset[7].onebody_features[:, t].numpy() - lo) / (hi - lo) - 1
        np.testing.assert_allclose(out[7].onebody_features[:, t].numpy(), expected, atol=1e-5)


@pytest.mark.parametrize(
    "pe_type", [p for p in PE_GROUPS if "hw" not in p.lower() and "res" not in p]
)
def test_builders_shapes(dataset, pe_type):
    d = normalize_dataset(dataset, PE_GROUPS[pe_type])[3]
    n = d.num_nodes
    node, edge, graph = (
        build_node_pe(d, pe_type),
        build_edge_pe(d, pe_type),
        build_graph_pe(d, pe_type),
    )
    assert node.shape[0] == n and edge.shape[0] == n * (n - 1) // 2 and graph.shape[0] == n
    assert graph.shape[1] == 2 * node.shape[1]
    assert np.isfinite(node).all() and np.isfinite(edge).all() and np.isfinite(graph).all()


def test_known_dims(dataset):
    d = normalize_dataset(dataset, PE_GROUPS["RRWP_FM_AFM"])[0]
    assert build_node_pe(d, "RRWP_FM_AFM").shape[1] == 20 + 2 * T
    assert build_graph_pe(d, "RRWP_FM_AFM").shape[1] == 40 + 4 * T
    d = normalize_dataset(dataset, PE_GROUPS["FM_stat"])[0]
    assert build_node_pe(d, "FM_stat").shape[1] == 4
    d = normalize_dataset(dataset, PE_GROUPS["RRWP_Laplacian_Ring_Geometric_Quantum"])[0]
    assert (
        build_node_pe(d, "RRWP_Laplacian_Ring_Geometric_Quantum").shape[1] == 20 + 4 + 6 + 2 + 2 * T
    )


def test_ring_membership():
    assert RING_SIZES == (3, 4, 5, 6, 7, 8)
    R = ring_membership_tensor(nx.cycle_graph(6))
    assert np.all(R[:, :, RING_SIZES.index(6)] == 1) and R.sum() == 36
    R = ring_membership_tensor(nx.Graph([(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)]))
    assert np.all(np.diagonal(R[:, :, 0]) == 1) and R[0, 3, 0] == 0


def test_controls_coincident_positions_are_finite():
    inter, dist = _pairwise_control_tensors(np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]))
    assert np.isfinite(inter).all() and inter[0, 1] == 0 and dist[0, 1] == 0
    np.testing.assert_allclose(inter[0, 2], C6_PULSER_N70 / 10.0**6)
