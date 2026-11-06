"""Encodings of the port vs the original internal implementation (bit-level check).

Needs results/downstream_gt/reference_features.pt, produced internally by
scripts/internal/downstream_reference_features.py, and the released data; skipped
otherwise.
"""

from pathlib import Path

import pytest
import torch

from qgf.downstream.gt.benchmark import build_splits, make_kfold_indices, transform_splits
from qgf.downstream.gt.data import load_mutag

REF = Path("results/downstream_gt/reference_features.pt")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not REF.exists(), reason="internal reference not available"),
]


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, weights_only=False)


def _compare(ref_cfg, out, tol=1e-5):
    worst = 0.0
    for split, r in ref_cfg.items():
        if split == "fold_test_idx":
            continue
        for i, ob, tb in zip(r["idx"], r["ob"], r["tb"]):
            mine = out[split][i]
            for a, b in (
                (ob, getattr(mine, "onebody_features", None)),
                (tb, getattr(mine, "twobody_features", None)),
            ):
                if a is None:
                    assert b is None
                    continue
                assert a.shape == b.shape
                worst = max(worst, (a - b).abs().max().item() / max(1.0, a.abs().max().item()))
    print(f"max rel diff {worst:.2e}")
    assert worst <= tol, worst
    return worst


def test_mutag_folds_match(ref):
    folds = make_kfold_indices(load_mutag([]))
    assert folds[3] == ref["mutag"]["rrwp"]["fold_test_idx"]


@pytest.mark.parametrize(
    "config",
    [
        "no_pe",
        "laplacian",
        "rrwp",
        "rrwp_laplacian",
        "af",
        "af_stat",
        "fm",
        "fm_stat",
        "af_fm",
        "af_mean",
        "af_mean_plus_4",
        "fm_mean_plus_5",
        "af_mean_fm_mean",
        "rrwp_af",
        "rrwp_af_stat_fm_stat",
        "rrwp_af_fm",
        "rrwp_lap_ring_geo",
        "rrwp_lap_ring_geo_af_fm",
        "rrwp_lap_ring_geo_af_fm_stat",
        "rrwp_lap_ring_geo_fmres_afmres",
        "rrwp_lap_ring_geo_afmres_stat",
    ],
)
def test_mutag_encodings(ref, config):
    raw = build_splits("mutag", config, fold=3)
    out, _, _ = transform_splits(config, raw)
    _compare(ref["mutag"][config], out)


@pytest.mark.parametrize(
    "config", ["rrwp", "af_stat", "rrwp_af_fm", "rrwp_fm_mean", "rrwp_lap_ring_geo_fmres_afmres"]
)
def test_zinc_encodings(ref, config):
    out, _, _ = transform_splits(config, build_splits("zinc", config, fold=None))
    _compare(ref["zinc"][config], out)
