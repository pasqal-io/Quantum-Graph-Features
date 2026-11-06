"""Logic tests of the downstream GT port on tiny synthetic graphs (CPU, fast)."""

import functools
import json
import operator

import pytest
import torch
from torch_geometric.data import Data

from qgf.downstream.gt import data as gt_data
from qgf.downstream.gt.benchmark import run_trial
from qgf.downstream.gt.pe_transforms import (
    CONFIGS,
    PEBenchmarkTransform,
    compute_rrwp,
    quantum_sources,
)


def ring(n: int, y, t: int = 20, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n)
    ei = torch.cat([torch.stack([src, (src + 1) % n]), torch.stack([(src + 1) % n, src])], 1)
    d = Data(
        x=torch.zeros(n, dtype=torch.long),
        edge_index=ei,
        edge_attr=torch.zeros(ei.shape[1], dtype=torch.long),
        y=y,
        pos=torch.randn(n, 2, generator=g) * 6,
        initial_id=torch.tensor([seed]),
    )
    for src_name in ("afm", "fm"):
        tb = torch.rand(t, n, n, generator=g)
        setattr(d, f"twobody_{src_name}", (tb + tb.transpose(1, 2)) / 2)
        setattr(d, f"onebody_{src_name}", torch.rand(t, n, generator=g))
    return d


def test_rrwp_cycle():
    ob, tb = compute_rrwp(ring(4, torch.tensor([0])), k=2)
    assert torch.allclose(
        tb[..., 0],
        torch.tensor([[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]),
    )
    assert torch.allclose(ob[:, 1], torch.full((4,), 0.5))  # return prob after 2 steps


def test_stats_fitted_on_train_only_and_time_major_input():
    train = [ring(n, torch.tensor([0]), seed=n) for n in (5, 6, 7)]
    other = ring(20, torch.tensor([0]), seed=99)  # N == T: orientation must not matter
    t = PEBenchmarkTransform("rrwp_af_stat_fm_stat", train)
    before = {k: [s.clone() for s in v] for k, v in t.stats.items()}
    out = t(other)
    assert all(torch.equal(a, b) for k in before for a, b in zip(before[k], t.stats[k]))
    assert out.onebody_features.shape == (20, 4 + 4 + 20)
    assert out.twobody_features.shape == (20, 20, 4 + 4 + 20)
    # AFM stat block = [mean, std, min, max] over time of the z-scored node features
    mean, std = t.stats[("afm", "stat")][:2]
    z = (other.onebody_afm.T - mean) / std  # [N, T]
    assert torch.allclose(out.onebody_features[:, 0], z.mean(-1))
    assert torch.allclose(out.onebody_features[:, 3], z.max(-1).values)
    assert "onebody_afm" not in out


def test_all_configs_build():
    train = [ring(n, torch.tensor([0]), seed=n) for n in (5, 6)]
    for cfg in CONFIGS:
        if set(quantum_sources(cfg)) - {"afm", "fm"}:  # the fixture carries these two only
            continue
        PEBenchmarkTransform(cfg, train)(train[0])


def test_attach_rescales_and_checks_alignment():
    base = [ring(3, torch.tensor([0]), seed=1)]
    other = [
        Data(
            onebody_features=torch.ones(2, 3) * 4,
            twobody_features=torch.ones(2, 3, 3) * 4,
            initial_id=torch.tensor([1]),
        )
    ]
    gt_data._attach(base, other, {"zz_scale_applied": 4.0}, "hw_afm")
    assert torch.equal(base[0].onebody_hw_afm, torch.ones(2, 3))
    other[0].initial_id = torch.tensor([2])
    with pytest.raises(ValueError):
        gt_data._attach(base, other, {"zz_scale_applied": 4.0}, "hw_afm")


def test_smoke_mutag_and_zinc_trials(tmp_path):
    mutag = [ring(4 + i % 5, torch.tensor([i % 2]), seed=i) for i in range(100)]
    r = run_trial(
        "mutag", "rrwp_af", seed=1, fold=0, output_dir=tmp_path / "m", max_epochs=1, graphs=mutag
    )
    assert (
        json.loads((tmp_path / "m/results.json").read_text())["test_f1_macro"] == r["test_f1_macro"]
    )
    assert (r["train_size"], r["val_size"], r["test_size"]) == (80, 10, 10)
    zinc = {
        k: [ring(4 + i % 3, torch.tensor([float(i)]), seed=i) for i in range(6)]
        for k in ("train", "val", "test")
    }
    for g in functools.reduce(operator.iadd, zinc.values(), []):
        g.x = g.x.unsqueeze(-1)  # ZINC atoms are [N, 1]
    r = run_trial("zinc", "fm_stat", seed=0, max_epochs=1, graphs=zinc)
    assert r["abs_pe_dim"] == 4 and r["test_mae"] >= 0


def _run_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "experiments/downstream_gt/run.py"
    spec = importlib.util.spec_from_file_location("downstream_gt_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finished_trial(tmp_path):
    run = _run_script()
    assert not run.finished(tmp_path)
    (tmp_path / "results.json").write_text(json.dumps({"test_mae": 0.1}))
    assert run.finished(tmp_path)
