import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from qgf.intrinsic.deepset_probe import (
    _entity_loader,
    _EntityDataset,
    _inner_split,
    evaluate_deepset_cv,
    evaluate_deepset_graph_cv,
)
from qgf.intrinsic.run import run_benchmark


def _toy(seed=42, n=30, cls=True):
    rng = np.random.RandomState(seed)
    pe = [rng.randn(20, 5) for _ in range(n)]
    tg = [rng.randint(0, 2, 20).astype(float) if cls else rng.randn(20) for _ in range(n)]
    return pe, tg


def test_fast_entity_loader_matches_dataloader_batches_and_rng():
    pe, tg = _toy()
    ds = _EntityDataset(pe, tg)
    torch.manual_seed(0)
    ref = [(x.clone(), y.clone()) for _ in range(2) for x, y in DataLoader(ds, 64, shuffle=True)]
    state_ref = torch.get_rng_state()
    torch.manual_seed(0)
    fast = [(x, y) for _ in range(2) for x, y in _entity_loader(ds, 64, shuffle=True)]
    assert torch.equal(state_ref, torch.get_rng_state())
    assert len(ref) == len(fast)
    for (x1, y1), (x2, y2) in zip(ref, fast):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_inner_split_partitions_train_graphs():
    train = np.arange(0, 100, 2)
    fit, val = _inner_split(train, seed=42, fold=0)
    assert len(val) == 5 and set(fit) | set(val) == set(train) and not set(fit) & set(val)


def test_entity_probe_scores_and_determinism():
    pe, tg = _toy()
    m1, s1, folds = evaluate_deepset_cv(pe, tg, classification=True, seed=99)
    m2, _, _ = evaluate_deepset_cv(pe, tg, classification=True, seed=99)
    assert 0 <= m1 <= 1 and s1 >= 0 and len(folds) == 5 and m1 == m2
    pe, tg = _toy(cls=False)
    m, s, _ = evaluate_deepset_cv(pe, tg, classification=False, seed=1, num_epochs=3)
    assert m > 0 and s >= 0


def test_graph_probe_with_fold_builder():
    rng = np.random.RandomState(0)
    pe = [rng.randn(rng.randint(3, 8), 4) for _ in range(25)]
    y = np.array([p.shape[0] for p in pe], dtype=float)
    seen = []

    def build(train_idx):
        seen.append(len(train_idx))
        return pe

    m, _, _ = evaluate_deepset_graph_cv(build, y, seed=0, num_epochs=3)
    assert m > 0 and seen == [20] * 5


def test_run_benchmark_smoke(dataset, tmp_path):
    for pe_type, task in [
        ("RRWP_FM_AFM", "degree_prediction"),
        ("FM_stat", "link_prediction"),
        ("RRWP_Laplacian_Ring_Geometric", "num_cycles"),
        ("random", "is_bridge_edge"),
    ]:
        out = tmp_path / pe_type / task
        r = run_benchmark(
            "toy", pe_type, task, str(out), cache_dir=str(tmp_path / "cache"), dataset=dataset
        )
        saved = json.loads((out / "result.json").read_text())
        assert saved["raw_value"] == r["raw_value"] and np.isfinite(r["raw_value"])
        assert len(saved["fold_scores"]) == 5
    assert (tmp_path / "cache" / "toy" / "targets" / "num_cycles.pkl").exists()
