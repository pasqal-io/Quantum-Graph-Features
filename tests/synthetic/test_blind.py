import numpy as np
import pytest

from qgf.synthetic.blind import balanced_subset, blind_pool
from qgf.synthetic.features import graph_features
from qgf.synthetic.ladder import ladder, unit_disk_positions


@pytest.fixture(scope="module")
def pool():
    return blind_pool(9, 11, [2, 3])  # 18-22 atoms, ~1 s


def test_blind_group_counts(pool):
    recs, gid = pool
    m = np.array([r["m"] for r in recs])
    k2 = gid[2] >= 0
    assert [int((k2 & (m == mm)).sum()) for mm in (9, 10, 11)] == [4, 20, 89]
    assert len(set(gid[2][k2 & (m == 11)])) == 19
    k3 = gid[3] >= 0  # first RRWP-3-blind graphs: 8 graphs in 2 groups at 22 atoms
    assert int(k3.sum()) == 8 and len(set(gid[3][k3])) == 2
    assert np.all(k2[k3])  # K = 3 blind implies K = 2 blind


def test_every_group_holds_both_labels(pool):
    recs, gid = pool
    y = np.array([r["y"] for r in recs])
    for g in set(gid[2][gid[2] >= 0]):
        assert set(y[gid[2] == g]) == {0, 1}


@pytest.mark.parametrize("K", [2, 3])
def test_pooled_rrwp_is_identical_inside_groups_but_ground_state_separates(pool, K):
    recs, gid = pool
    for g in set(gid[K][gid[K] >= 0]):
        idx = np.nonzero(gid[K] == g)[0]
        F = []
        for i in idx:
            m, cr = recs[i]["m"], recs[i]["cross"]
            F.append(graph_features(m, cr, unit_disk_positions(ladder(m, cr))))
        for fam in (f"rrwp{K}", "ring", "n"):
            for f in F[1:]:
                assert np.array_equal(f[fam], F[0][fam]), (K, g, fam)
        for f, i in zip(F, idx):  # pinned <=> ground-state correlations vanish
            assert (f["gs_max"][0] == 0.0) == (recs[i]["y"] == 1)


def test_balanced_subset_balances_every_group():
    y = np.array([1, 0, 0, 1, 1, 1, 0, 0])
    g = np.array([0, 0, 0, 1, 1, 1, 1, -1])
    sel = balanced_subset(y, g)
    assert -1 not in g[sel]
    for grp in (0, 1):
        ys = y[sel][g[sel] == grp]
        assert (ys == 1).sum() == (ys == 0).sum() == {0: 1, 1: 1}[grp]
    assert np.array_equal(sel, balanced_subset(y, g))  # seeded
