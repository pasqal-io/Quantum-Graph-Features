"""Encodings of the port vs those fed to the original GNN runs.

Needs results/downstream_gnns/reference_features.pt, produced internally by
scripts/internal/downstream_gnns_reference_features.py, and the released data; skipped
otherwise.
"""

from pathlib import Path

import pytest
import torch

from qgf.downstream.gt.benchmark import build_splits, transform_splits

REF = Path("results/downstream_gnns/reference_features.pt")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not REF.exists(), reason="internal reference not available"),
]


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, weights_only=False)


def max_rel_diff(ref_cfg: dict, out: dict, part: str) -> float:
    """Largest relative difference of the ``ob`` or ``tb`` blocks of the dumped graphs."""
    attr = {"ob": "onebody_features", "tb": "twobody_features"}[part]
    worst = 0.0
    for split, r in ref_cfg.items():
        for i, gid, a in zip(r["idx"], r["initial_id"], r[part]):
            mine = out[split][i]
            assert int(mine.initial_id.flatten()[0]) == gid
            b = getattr(mine, attr, None)
            if a is None:
                assert b is None
                continue
            assert a.shape == b.shape, (split, i, a.shape, b.shape)
            worst = max(worst, (a - b).abs().max().item() / max(1.0, a.abs().max().item()))
    return worst


MUTAG = [
    "no_pe",
    "laplacian",
    "rrwp",
    "rrwp_laplacian",
    "af",
    "af_stat",
    "fm",
    "fm_stat",
    "af_fm",
    "rrwp_af",
    "rrwp_fm_stat",
    "rrwp_af_fm",
    "rrwp_af_stat_fm_stat",
    "rrwp_lap_ring_geom",
    "rrwp_lap_ring_geom_af_fm",
]


@pytest.mark.parametrize("config", MUTAG)
def test_mutag_encodings(ref, config):
    out, _, _ = transform_splits(config, build_splits("mutag", config, fold=ref["fold"]))
    for part in ("ob", "tb"):
        assert max_rel_diff(ref["mutag"][config], out, part) <= 1e-5


@pytest.mark.parametrize("config", ["hw_afm", "rrwp_hw_afm_stat"])
def test_mutag_hardware_encodings(ref, config):
    """The released hardware set corrects the one-body (diagonal) estimator; the two-body
    block is unchanged."""
    out, _, _ = transform_splits(config, build_splits("mutag", config, fold=ref["fold"]))
    assert max_rel_diff(ref["mutag"][config], out, "tb") <= 1e-5


@pytest.mark.parametrize("config", ["rrwp", "af_stat", "rrwp_af_fm", "rrwp_lap_ring_geom_af_fm"])
def test_zinc_encodings(ref, config):
    out, _, _ = transform_splits(config, build_splits("zinc", config, fold=None))
    for part in ("ob", "tb"):
        assert max_rel_diff(ref["zinc"][config], out, part) <= 1e-5
