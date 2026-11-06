import numpy as np
import pytest

from qgf.generate.correlations import (
    correlation_from_counts,
    jackknife_correlation_from_counts,
    split_features,
)
from qgf.generate.raw import decode_counts, encode_counts, load_raw, save_raw

COUNTS = {"110": 3, "011": 2, "000": 4, "101": 1, "111": 2}


def _loop_reference(counts):
    """Literal port of the original loop implementation."""
    n = len(next(iter(counts)))
    avg_z, avg_zz, total = np.zeros(n), np.zeros((n, n)), sum(counts.values())
    for s, c in counts.items():
        occ = np.array([1 if ch == "1" else 0 for ch in s])
        avg_z += occ * c
        avg_zz += np.outer(occ, occ) * c
    avg_z /= total
    avg_zz /= total
    return avg_zz - np.outer(avg_z, avg_z)


def _jackknife_reference(counts, g):
    """Leave-one-shot-out jackknife over the expanded shot list, for pair (0, 1)."""
    shots = np.array([[ch == "1" for ch in s] for s, c in counts.items() for _ in range(c)], float)
    s = len(shots)
    loo = []
    for k in range(s):
        rest = np.delete(shots, k, axis=0)
        loo.append(g(np.mean(rest[:, 0] * rest[:, 1]), rest[:, 0].mean(), rest[:, 1].mean()))
    loo = np.array(loo)
    full = g(np.mean(shots[:, 0] * shots[:, 1]), shots[:, 0].mean(), shots[:, 1].mean())
    est = s * full - (s - 1) * loo.mean()
    err = np.sqrt(np.sum((loo - loo.mean()) ** 2) * (s - 1) / s)
    return est, err


def test_plugin_matches_loop():
    np.testing.assert_allclose(correlation_from_counts(COUNTS), _loop_reference(COUNTS), atol=1e-15)


def test_jackknife_without_readout_matches_shot_level_jackknife():
    c, err = jackknife_correlation_from_counts(COUNTS, epsilons=None)
    est, e = _jackknife_reference(COUNTS, lambda nn, ni, nj: nn - ni * nj)
    assert c[0, 1] == pytest.approx(est) and err[0, 1] == pytest.approx(e)
    # Jackknife bias correction of a covariance gives the unbiased (n-1) estimator.
    s = sum(COUNTS.values())
    np.testing.assert_allclose(c, correlation_from_counts(COUNTS) * s / (s - 1), atol=1e-12)


def test_readout_correction_inverts_confusion_on_exact_moments():
    # Two independent atoms with p=0.3 and p=0.6 -> true C_01 = 0, C_00 = 0.21.
    eps, epsp = 0.025, 0.1
    p = np.array([0.3, 0.6])
    q = p * (1 - epsp) + (1 - p) * eps  # observed marginals
    counts = {}
    for a in "01":
        for b in "01":
            pa = q[0] if a == "1" else 1 - q[0]
            pb = q[1] if b == "1" else 1 - q[1]
            counts[a + b] = round(1e6 * pa * pb)
    c, _ = jackknife_correlation_from_counts(counts, epsilons=(eps, epsp))
    assert c[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 0] == pytest.approx(0.21, abs=1e-4)
    assert c[1, 1] == pytest.approx(0.24, abs=1e-4)


def test_split_features_zz_scale_and_zero_diagonal():
    corr = np.stack([correlation_from_counts(COUNTS)] * 2)
    ob, tb = split_features(corr)
    np.testing.assert_allclose(ob[0], 4 * np.diag(corr[0]))
    assert np.all(np.diagonal(tb, axis1=1, axis2=2) == 0)
    np.testing.assert_allclose(tb[1, 0, 1], 4 * corr[1, 0, 1])


def test_raw_roundtrip(tmp_path):
    bits, counts = encode_counts({"100": 2, "001": 1})
    assert bits.tolist() == [1, 4]  # bit i = atom i
    assert decode_counts(bits, counts, 3) == {"100": 2, "001": 1}
    records = [{10: COUNTS, 20: {"001": 5}}, {10: {"11": 1}}]
    save_raw(tmp_path / "r.npz", records, [7, 9], [3, 2], [10, 20], {"name": "x"})
    raw = load_raw(tmp_path / "r.npz")
    assert raw.counts(0, 0) == COUNTS and raw.counts(1, 1) == {}
    assert raw.index_of(9) == 1 and raw.meta["name"] == "x"
    b, w = raw.bit_matrix(0, 1)
    assert b.tolist() == [[0, 0, 1]] and w.tolist() == [5]
