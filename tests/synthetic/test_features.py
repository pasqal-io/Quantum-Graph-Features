import networkx as nx
import numpy as np

from qgf.synthetic.features import (
    COMBOS,
    add_combos,
    graph_features,
    pool1,
    pool2,
    quench_features,
)
from qgf.synthetic.ladder import ladder, unit_disk_positions


def test_pooling_is_permutation_invariant():
    rng = np.random.default_rng(0)
    n = 12
    X, T = rng.random((n, 3)), rng.random((n, n, 2))
    A = np.triu((rng.random((n, n)) < 0.3).astype(float), 1)
    A = A + A.T
    p = rng.permutation(n)
    assert np.allclose(pool1(X[p]), pool1(X))
    assert np.allclose(pool2(T[p][:, p], A[p][:, p]), pool2(T, A))


def test_quench_features_transpose_time_major_explicitly_when_n_equals_t():
    # a 10-rung ladder has N = 20 atoms = T = 20 times: the layout cannot come from shapes
    m, T = 10, 20
    A = nx.to_numpy_array(ladder(m, (3,)), nodelist=range(2 * m))
    rng = np.random.default_rng(1)
    occ = rng.random((T, 2 * m))  # [T, N]
    corr = rng.random((T, 2 * m, 2 * m))
    corr = corr + corr.transpose(0, 2, 1)
    expected = np.concatenate([pool1(occ.T), pool2(np.moveaxis(corr, 0, -1), A)])
    assert np.array_equal(quench_features(occ, corr, A), expected)
    assert not np.allclose(quench_features(occ.T, corr, A), expected)  # node-major is wrong


def test_graph_features_families_and_dimensions():
    m, cr = 9, (1, 4)
    pos = unit_disk_positions(ladder(m, cr))
    T, n = 20, 2 * m
    q = (np.full((T, n), 0.3), np.zeros((T, n, n)))
    F = add_combos(graph_features(m, cr, pos, q, q))
    dims = {k: v.shape[0] for k, v in F.items()}
    assert dims["n"] == 1 and dims["rrwp2"] == 60 and dims["rrwp20"] == 600
    assert dims["geom"] == 110 and dims["lap"] == 60 and dims["ring"] == 66
    assert dims["afm"] == dims["fm"] == 600 and dims["gs"] == 30
    assert set(COMBOS) <= set(F)
    assert dims["rrwp20+lap+ring+geom+afm+fm"] == 600 + 60 + 66 + 110 + 1200
