"""Logic tests of the downstream GNN port on tiny synthetic graphs (CPU, fast)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from qgf.downstream.gnns.benchmark import build_model, make_loader, run_trial
from qgf.downstream.gnns.model import (
    ALL_MODELS,
    GNNWithPE,
    LinearCorrelationEncoder,
    shift_per_graph,
)
from qgf.downstream.gt.pe_transforms import PEBenchmarkTransform

ROOT = Path(__file__).parents[2]


def ring(n: int, y: int, t: int = 20, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n)
    ei = torch.cat([torch.stack([src, (src + 1) % n]), torch.stack([(src + 1) % n, src])], 1)
    d = Data(
        x=torch.randint(0, 7, (n,), generator=g),
        edge_index=ei,
        edge_attr=torch.randint(0, 4, (ei.shape[1],), generator=g),
        y=torch.tensor([y]),
        pos=torch.randn(n, 2, generator=g) * 6,
        initial_id=torch.tensor([seed]),
    )
    for name in ("afm", "fm"):
        tb = torch.rand(t, n, n, generator=g)
        setattr(d, f"twobody_{name}", (tb + tb.transpose(1, 2)) / 2)
        setattr(d, f"onebody_{name}", torch.rand(t, n, generator=g))
    return d


def _script(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(config: str, graphs: list[Data]):
    t = PEBenchmarkTransform(config, graphs)
    out = [t(d) for d in graphs]
    ob = getattr(out[0], "onebody_features", None)
    tb = getattr(out[0], "twobody_features", None)
    abs_dim = 0 if ob is None else ob.shape[1]
    rel_dim = 0 if tb is None else tb.shape[2]
    return next(iter(make_loader(out, len(out), shuffle=False))), abs_dim, rel_dim


@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("config", ["no_pe", "rrwp_af_fm"])
def test_forward_shapes(model_type, config):
    graphs = [ring(n, n % 2, seed=n) for n in (5, 6, 7)]
    batch, abs_dim, rel_dim = _batch(config, graphs)
    for dataset, out_dim in (("mutag", 2), ("zinc", 1)):
        model = build_model(dataset, model_type, abs_dim, rel_dim)
        assert model(batch).shape == (3, out_dim)


def test_edge_inputs_follow_the_backbone():
    graphs = [ring(n, 0, seed=n) for n in (5, 6)]
    batch, abs_dim, rel_dim = _batch("rrwp_af", graphs)
    gcn = build_model("mutag", "GCN", abs_dim, rel_dim)
    assert gcn.twobody_encoder.coefficients.shape == (rel_dim + 50,)
    assert gcn.edge_weight_shift  # GCN only
    gat = build_model("mutag", "GAT", abs_dim, rel_dim)
    assert not hasattr(gat, "twobody_encoder") and not gat.edge_weight_shift
    pair = gat._per_graph(batch, batch.twobody_features)
    assert pair.shape == (batch.edge_index.shape[1], rel_dim + 50)
    # the two-body part of each edge attribute is the encoding of that node pair
    src, dst = graphs[0].edge_index[:, 0]
    assert torch.equal(pair[0, :rel_dim], batch.twobody_features[0][src, dst])


def test_gcn_edge_weights_are_shifted_per_graph():
    w = torch.tensor([0.5, 2.0, -1.0, 3.0, 4.0])
    gid = torch.tensor([0, 0, 1, 1, 1])
    out = shift_per_graph(w, gid, 1e-3)
    assert torch.allclose(out, torch.tensor([1e-3, 1.501, 1e-3, 4.001, 5.001]))


def test_linear_correlation_encoder_is_a_weighted_sum():
    enc = LinearCorrelationEncoder(3)
    m = torch.rand(3, 4, 4)
    assert torch.allclose(enc(m), m.mean(0))


def test_construction_order_matches_the_original():
    torch.manual_seed(0)
    model = GNNWithPE("GCN", 7, 4, [8, 8], [8], abs_pe_dim=2, rel_pe_dim=3)
    names = [n for n, _ in model.named_parameters()]
    assert names[:3] == [
        "node_embedding.weight",
        "twobody_encoder.coefficients",
        "bond_embedding.weight",
    ]


def mutag_like() -> list[Data]:
    """80 tiny graphs: enough for 10 folds stratified by label x size quartile."""
    return [ring(5 + i % 20, (i // 20) % 2, t=4, seed=i) for i in range(80)]


def test_run_trial_smoke(tmp_path):
    graphs = mutag_like()
    r = run_trial(
        "mutag",
        "GINE",
        "rrwp_af_stat",
        seed=1,
        fold=0,
        output_dir=tmp_path,
        max_epochs=2,
        graphs=graphs,
    )
    stored = json.loads((tmp_path / "results.json").read_text())
    assert stored["test_f1_macro"] == r["test_f1_macro"]
    assert r["train_size"] + r["val_size"] + r["test_size"] == 80
    assert r["abs_pe_dim"] == 24 and r["model_config"]["gnn_layers"] == [64] * 4


def test_run_trial_is_seeded():
    graphs = mutag_like()
    a = run_trial("mutag", "GCN", "af", seed=2, fold=1, max_epochs=2, graphs=graphs)
    b = run_trial("mutag", "GCN", "af", seed=2, fold=1, max_epochs=2, graphs=graphs)
    assert a["test_f1_macro"] == b["test_f1_macro"] and a["best_epoch"] == b["best_epoch"]


def test_analyze_aggregates_over_trials(tmp_path):
    for fold in range(2):
        for seed in (1, 2):
            d = tmp_path / "mutag" / "GCN" / "rrwp" / f"fold_{fold}" / f"seed_{seed}"
            d.mkdir(parents=True)
            r = {
                "model_type": "GCN",
                "config": "rrwp",
                "fold": fold,
                "seed": seed,
                "test_f1_macro": 0.8 + 0.1 * fold,
            }
            (d / "results.json").write_text(json.dumps(r))
    exp = tmp_path / "exp"
    exp.mkdir()
    pd.DataFrame([{"model": "GCN", "config": "rrwp", "mean": 0.85, "std": 0.1}]).to_csv(
        exp / "mutag_summary.csv", index=False
    )
    mod = _script("experiments/downstream_gnns/analyze.py")
    mod.main(["--results", str(tmp_path), "--datasets", "mutag", "--expected", str(exp)])
    s = pd.read_csv(tmp_path / "mutag_summary.csv")
    assert s.loc[0, "mean"] == pytest.approx(0.85) and s.loc[0, "n_trials"] == 4
    assert s.loc[0, "std"] == pytest.approx(pd.Series([0.8, 0.8, 0.9, 0.9]).std())


def test_analyze_zinc_trials_and_claims(tmp_path):
    """ZINC: per-trial CSV in the columns of expected/zinc_trials.csv; claims pick the lowest
    MAE as the column best."""
    maes = {"rrwp": 0.12, "rrwp_af_stat_fm_stat": 0.11, "no_pe": 0.30}
    for config, mae in maes.items():
        for seed in (0, 1):
            d = tmp_path / "zinc" / "GINE" / config / f"seed_{seed}"
            d.mkdir(parents=True)
            r = {"model_type": "GINE", "config": config, "fold": None, "seed": seed}
            r |= {"test_mae": mae + 0.01 * seed, "abs_pe_dim": 4, "rel_pe_dim": 4}
            (d / "results.json").write_text(json.dumps(r))
    exp = tmp_path / "exp"
    exp.mkdir()
    ref = [
        {"model": "GINE", "config": c, "mean": m, "std": 0.01, "n_trials": 2}
        for c, m in maes.items()
    ]
    pd.DataFrame(ref).to_csv(exp / "zinc_summary.csv", index=False)
    mod = _script("experiments/downstream_gnns/analyze.py")
    mod.main(["--results", str(tmp_path), "--datasets", "zinc", "--expected", str(exp)])
    trials = pd.read_csv(tmp_path / "zinc_trials.csv")
    expected_cols = pd.read_csv(
        ROOT / "experiments/downstream_gnns/expected/zinc_trials.csv", nrows=1
    ).columns
    assert list(trials.columns) == list(expected_cols) and len(trials) == 6
    assert (trials.fold == -1).all() and (trials.source == "rerun").all()
    c = pd.read_csv(tmp_path / "zinc_claims.csv").set_index("src")
    assert (c.table3_best == "rrwp_af_stat_fm_stat").all() and c.best_is_hybrid.all()
    assert c.loc["rerun", "champion_gap"] == pytest.approx(-0.01)
    cmp = pd.read_csv(tmp_path / "zinc_comparison.csv")  # re-run means = ref + 0.005
    assert cmp.delta_over_std.to_numpy() == pytest.approx([0.5] * 3)
    se = (0.01**2 / 2 / 2 + 0.01**2 / 2) ** 0.5  # re-run std (ddof 1) 0.01 / sqrt(2), n 2
    assert cmp.delta_over_se.to_numpy() == pytest.approx([0.005 / se] * 3, rel=1e-3)


def test_run_grid_skips_finished_trials(tmp_path, monkeypatch):
    mod = _script("experiments/downstream_gnns/run.py")
    calls = []
    monkeypatch.setattr(mod, "run_trial", lambda *a, **k: calls.append(a) or {"test_mae": 0.0})
    done = mod.trial_dir(tmp_path, "zinc", "GCN", "rrwp", None, 0)
    done.mkdir(parents=True)
    (done / "results.json").write_text("{}")
    assert (
        mod.main(
            ["--dataset", "zinc", "--models", "GCN", "--configs", "rrwp", "--out", str(tmp_path)]
        )
        == 0
    )
    assert [a[3] for a in calls] == [1, 2, 3, 4]  # seeds 1-4 ran, seed 0 was skipped
    assert len(mod.GRIDS["mutag"]) == 22 and "hw_afm" not in mod.GRIDS["zinc"]
    calls.clear()
    assert mod.main(["--dataset", "zinc", "--configs", "validation", "--out", str(tmp_path)]) == 0
    assert len(calls) == 2 * 5 and {a[1] for a in calls} == {"GINE", "GAT"}


def test_gnn_strong_baseline_encodings():
    """GNN geometry baseline: RRWP 20 + Laplacian 4 + cycle-basis rings 6 (unscaled) +
    register geometry 5 / 3 (z-scored), as in the original GNN runs (35 / 33)."""
    graphs = [ring(n, 0, seed=n) for n in (5, 6, 7)]
    t = PEBenchmarkTransform("rrwp_lap_ring_geom", graphs)
    out = t(graphs[0])
    assert out.onebody_features.shape == (5, 35) and out.twobody_features.shape == (5, 5, 33)
    ring5 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 5)  # a 5-cycle, unscaled
    assert torch.equal(out.onebody_features[:, 24:30], ring5)
    assert ("ring_cb", "raw") not in t.stats
    with_q = PEBenchmarkTransform("rrwp_lap_ring_geom_af_fm", graphs)(graphs[0])
    assert with_q.onebody_features.shape == (5, 75) and with_q.twobody_features.shape == (5, 5, 73)
    assert torch.equal(with_q.onebody_features[:, 40:], out.onebody_features)  # quantum first
