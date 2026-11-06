from itertools import combinations

import numpy as np
import pytest

from qgf.synthetic.ladder import (
    A_UM,
    gs_correlations,
    gs_correlations_dp,
    ladder,
    mis_bruteforce,
    parse_crossings,
    unit_disk_positions,
    verify_unit_disk,
)


@pytest.mark.parametrize("m", [2, 3, 5, 6])
def test_dp_matches_bruteforce_on_every_crossing_subset(m):
    for r in range(m):
        for cr in combinations(range(m - 1), r):
            G = ladder(m, cr)
            occ_b, C_b = gs_correlations(G)
            occ, C, count = gs_correlations_dp(G)
            assert count == len(mis_bruteforce(G)[1])
            np.testing.assert_allclose(occ, occ_b, atol=1e-12)
            np.testing.assert_allclose(C, C_b, atol=1e-12)


def test_pinned_is_a_product_state_frustrated_is_not():
    _, C, count = gs_correlations_dp(ladder(200, (20, 60, 140)))  # same parity
    assert count == 1 and np.abs(C).max() == 0.0
    _, C, count = gs_correlations_dp(ladder(200, (20, 61)))  # conflicting parity
    assert count > 1 and np.abs(C).max() > 0.2


def test_dp_handles_huge_degeneracy():
    assert gs_correlations_dp(ladder(400, (30, 97, 170, 233, 301, 360)))[2] == 17144


def test_register_is_unit_disk_with_min_distance_r_nn():
    for m, cr in [(9, (0,)), (11, (2, 3, 7)), (13, tuple(range(12)))]:
        G = ladder(m, cr)
        pos = unit_disk_positions(G)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[np.triu_indices(2 * m, 1)]
        assert pos.shape == (2 * m, 2) and d.min() == pytest.approx(A_UM)
        ok, missing, extra, gap = verify_unit_disk(G)
        assert ok and missing == extra == 0 and gap >= np.sqrt(2) - 1e-9


def test_builder_and_parsing():
    G = ladder(4, (1,))
    assert G.number_of_nodes() == 8 and G.number_of_edges() == 3 * 4 - 2 + 1
    assert G.has_edge(1, 4 + 2)
    assert parse_crossings("2,5") == (2, 5) and parse_crossings("") == ()
    with pytest.raises(ValueError):
        ladder(4, (3,))
