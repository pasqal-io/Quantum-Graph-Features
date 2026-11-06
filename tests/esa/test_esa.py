"""Logic tests of the ESA port on tiny synthetic graphs (CPU, fast)."""

import json
import math

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

pytest.importorskip("pytorch_lightning")  # the esa extra

from qgf.esa.benchmark import esa_graph, run_trial, rw_landing_probs  # noqa: E402
from qgf.esa.layers import MASK_FILL, admin_residual, edge_adjacency_bias  # noqa: E402
from qgf.esa.model import ESAClassifier  # noqa: E402
from qgf.esa.trainer import binary_metrics_pt  # noqa: E402

SMALL = {"hidden_dim": 16, "num_heads": 2, "num_layers": 4, "graph_dim": 8, "mlp_hidden_size": 8}


def ring(n: int, y: int, t: int = 4, seed: int = 0) -> Data:
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


def test_admin_residual_matches_admin_torch_default():
    res = admin_residual(16)
    omega = math.sqrt(17 / math.log(17) - 1)
    x, fx = torch.ones(3), torch.full((3,), 2.0)
    assert torch.allclose(res(x, fx), x * omega + fx)


def test_edge_adjacency_bias_two_graphs():
    # graph 0: path 0-1-2 (edges 0->1, 1->2); graph 1: single edge 3->4
    ei = torch.tensor([[0, 1, 3], [1, 2, 4]])
    bias = edge_adjacency_bias(ei, torch.tensor([0, 0, 0, 1, 1]), batch_size=2, max_items=8)
    assert bias.shape == (2, 1, 8, 8)
    assert bias[0, 0, 0, 1] == 0 and bias[0, 0, 1, 0] == 0  # share node 1
    assert bias[0, 0, 0, 0] == MASK_FILL  # an edge does not attend to itself
    assert (bias[1, 0] == MASK_FILL).all()  # lone edge: no neighbour
    assert (bias[0, 0, 2:] == MASK_FILL).all()  # padding


def test_hw_afm_3t_selects_305_512_4000_ns():
    from qgf.downstream.gt import data as gt_data
    from qgf.downstream.gt.pe_transforms import quantum_sources

    assert quantum_sources("hw_afm_3t_stat") == ["hw_afm_3t"]
    times = [305, 512, 715, 4000]
    t = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(4, 1)  # value = 4 C per time, C <= 0.4
    other = [
        Data(
            onebody_features=(t * torch.ones(4, 3)) * 4,
            twobody_features=(t.view(4, 1, 1) * torch.ones(4, 3, 3)) * -4,
            initial_id=torch.tensor([7]),
        )
    ]
    base = [Data(initial_id=torch.tensor([7]))]
    meta = {"zz_scale_applied": 4.0, "times_ns": times}
    gt_data._attach(base, other, meta, "hw_afm_3t")
    ob, tb = base[0].onebody_hw_afm_3t, base[0].twobody_hw_afm_3t
    assert ob.shape == (3, 3) and tb.shape == (3, 3, 3)  # 715 ns dropped
    # ZZ convention kept and clipped to [-1, 1], as the loader of the paper runs
    assert torch.equal(ob[:, 0], torch.tensor([0.4, 0.8, 1.0]))
    assert torch.equal(tb[:, 0, 1], torch.tensor([-0.4, -0.8, -1.0]))
    gt_data._attach(base, other, meta, "hw_afm")  # the released set: all 4 times, C
    assert base[0].onebody_hw_afm.shape == (4, 3)
    assert torch.allclose(base[0].onebody_hw_afm[:, 0], t.view(-1))


def test_rw_landing_probs_cycle():
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    rw = rw_landing_probs(ei, 4, steps=4)
    assert torch.allclose(rw[:, 0], torch.zeros(4)) and torch.allclose(
        rw[:, 1], torch.full((4,), 0.5)
    )


def test_binary_metrics_follow_torchmetrics_logit_rule():
    y = torch.tensor([1, 0, 1, 0])
    logits = torch.tensor([2.0, -1.0, 0.3, -3.0])  # outside [0, 1]: sigmoid applied
    assert float(binary_metrics_pt(y, logits)[3]) == 1.0
    probs = torch.tensor([0.9, 0.1, 0.4, 0.2])  # inside [0, 1]: thresholded as is
    assert float(binary_metrics_pt(y, probs)[3]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("fusion", ["concat", "gated"])
def test_forward_and_gate(fusion):
    torch.manual_seed(0)
    from qgf.downstream.gt.pe_transforms import PEBenchmarkTransform

    graphs = [ring(n, n % 2, seed=n) for n in (5, 6, 7)]
    t = PEBenchmarkTransform("af_stat", graphs)
    data = [esa_graph(t(d)) for d in graphs]
    assert data[0].x.shape == (5, 7 + 6 + 4) and data[0].edge_attr.shape == (10, 4 + 4)
    model = ESAClassifier(
        17, 8, set_max_items=14, fusion=fusion, quantum_node_dim=4, quantum_edge_dim=4, **SMALL
    )
    out = model(next(iter(DataLoader(data, batch_size=3))))
    assert out.shape == (3,)
    if fusion == "gated":
        assert 0.8 < model.last_gate_mean < 0.95  # bias 2 at init: sigmoid(2) ~ 0.88
    else:
        assert model.last_gate_mean is None


def test_run_trial_smoke(tmp_path):
    graphs = [ring(5 + i % 20, (i // 20) % 2, seed=i) for i in range(80)]
    r = run_trial(
        "af_fm",
        "gated",
        seed=1,
        fold=0,
        output_dir=tmp_path,
        max_epochs=2,
        graphs=graphs,
        model_kwargs=SMALL,
    )
    stored = json.loads((tmp_path / "results.json").read_text())
    assert stored["test_f1"] == r["test_f1"] and r["fusion"] == "gated"
    assert r["abs_pe_dim"] == 8 and r["gate_mean"] is not None
    no_q = run_trial(
        "no_pe", "gated", seed=1, fold=0, max_epochs=1, graphs=graphs, model_kwargs=SMALL
    )
    assert no_q["fusion"] == "concat"  # nothing to gate


def test_analyze_counts_identical_trials(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).parents[2] / "experiments" / "esa" / "analyze.py"
    spec = importlib.util.spec_from_file_location("esa_analyze", path)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    ref_rows = []
    for fold in range(2):
        r = {"fusion": "gated", "config": "hw_afm_3t", "fold": fold, "seed": 1,
             "test_f1": 0.96 - fold * 0.1, "test_accuracy": 0.9, "test_auc": 0.95,
             "test_mcc": 0.8, "gate_mean": 0.7890625, "abs_pe_dim": 3}  # fmt: skip
        d = tmp_path / "res" / "gated" / "hw_afm_3t" / f"fold_{fold}" / "seed_1"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(r))
        ref_rows.append({k.removeprefix("test_"): v for k, v in r.items()})
    ref_rows[1]["f1"] += 1e-4  # one trial differs
    exp = tmp_path / "exp"
    exp.mkdir()
    pd.DataFrame(ref_rows).to_csv(exp / "mutag_trials.csv", index=False)
    analyze.aggregate(pd.DataFrame(ref_rows)).to_csv(exp / "mutag_summary.csv", index=False)
    trials_out = tmp_path / "trials.csv"
    assert analyze.main(["--results", str(tmp_path / "res"), "--expected", str(exp),
                         "--trials-out", str(trials_out)]) == 0  # fmt: skip
    assert "identical trials: 1/2" in capsys.readouterr().out
    assert len(pd.read_csv(trials_out)) == 2
