"""Fig. 2 toy reconstruction (experiments/reconstruction/fig2.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qgf.data import DEFAULT_ROOT

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "experiments/reconstruction/expected"


def _fig2():
    path = ROOT / "experiments/reconstruction/fig2.py"
    spec = importlib.util.spec_from_file_location("fig2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _path(n: int) -> np.ndarray:
    a = np.zeros((n, n))
    a[np.arange(n - 1), np.arange(1, n)] = 1.0
    return a + a.T


def test_walk_and_rrwp_features():
    fig2 = _fig2()
    a = _path(4)
    x = fig2.walk_features(a, 2)  # pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    assert x.shape == (6, 3)
    np.testing.assert_array_equal(x[:, 0], 1.0)
    np.testing.assert_array_equal(x[:, 1], [1, 0, 0, 1, 0, 1])  # A_ij
    np.testing.assert_array_equal(x[:, 2], [0, 1, 0, 0, 1, 0])  # (A^2)_ij
    r = fig2.rrwp_features(a, 1)
    np.testing.assert_allclose(r[:, 1], [1.0, 0, 0, 0.5, 0, 0.5])  # M = D^-1 A, row i


def test_pooled_r2_is_one_for_a_linear_target():
    fig2 = _fig2()
    rng = np.random.default_rng(0)
    x = np.c_[np.ones(30), rng.normal(size=(30, 2))]
    y = np.stack([x @ [0.1, 2.0, -1.0], rng.normal(size=30)])
    r2 = fig2.pooled_r2(x, y)
    assert r2[0] == pytest.approx(1.0)
    assert r2[1] < 0.5


def test_expected_files_cover_the_figure():
    for name in ("fig2_r2_published.csv", "fig2_r2_paper.csv"):
        df = pd.read_csv(EXPECTED / name)
        assert len(df) == 2 * 2 * 9 * 20  # pulse x family x K x time
        assert set(df.K) >= {1, 3, 7, 21}


@pytest.mark.slow
@pytest.mark.skipif(not (DEFAULT_ROOT / "mutag_afm.pt").exists(), reason="needs mutag_afm")
def test_reproduces_published_curves():
    fig2 = _fig2()
    ids, adj = fig2.toy_graphs()
    assert ids == [75, 115]
    for quench in ("published", "paper"):
        got = fig2.recovery_curves(adj, quench)
        ref = pd.read_csv(EXPECTED / f"fig2_r2_{quench}.csv")
        got["t_ns"], ref["t_ns"] = got.t_ns.round(4), ref.t_ns.round(4)
        m = got.merge(ref, on=["pulse", "family", "K", "t_ns"], suffixes=("", "_ref"))
        assert len(m) == len(ref)
        np.testing.assert_allclose(m.r2, m.r2_ref, atol=1e-6)
