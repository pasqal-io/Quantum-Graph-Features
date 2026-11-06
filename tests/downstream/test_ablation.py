"""Logic tests of the ZINC_tiny ablation port and its analysis (CPU, fast)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from qgf.downstream.gnns.ablation import normalisation, run_trial, source_arrays, transform


def graph(n: int, seed: int, t: int = 5) -> Data:
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n)
    ei = torch.cat([torch.stack([src, (src + 1) % n]), torch.stack([(src + 1) % n, src])], 1)
    occ = torch.rand(t, n, generator=g)
    c = torch.rand(t, n, n, generator=g) * 0.1
    c = (c + c.transpose(1, 2)) / 2
    return Data(x=torch.randint(0, 95, (n, 1), generator=g), edge_index=ei,
                edge_attr=torch.randint(0, 4, (ei.shape[1],), generator=g),
                y=torch.rand(1, generator=g), ob=occ, tb=c)  # fmt: skip


def splits():
    return {k: [graph(5 + i % 4, seed=i + 100 * j) for i in range(8)]
            for j, k in enumerate(("train", "val", "test"))}  # fmt: skip


def test_source_arrays_restore_occupation_and_diagonal():
    n, t = 4, 3
    occ = torch.rand(t, n)
    corr = torch.rand(t, n, n)
    corr = (corr + corr.transpose(1, 2)) / 2
    idx = torch.arange(n)
    corr[:, idx, idx] = occ * (1 - occ)
    released = Data(occupation=occ, onebody_features=4 * occ * (1 - occ),
                    twobody_features=4 * corr * (1 - torch.eye(n)))  # fmt: skip
    ob, tb = source_arrays(released, 4.0)
    assert torch.allclose(ob, occ) and torch.allclose(tb, corr)


def test_normalisation_uses_training_split_and_full_matrix():
    s = splits()
    ob_mean, ob_std, tb_mean, tb_std = normalisation(s["train"])
    tb_all = torch.cat([g.tb.permute(1, 2, 0).reshape(-1, 5) for g in s["train"]])
    assert torch.allclose(tb_mean, tb_all.mean(0))  # diagonal included
    out, abs_dim, rel_dim = transform(s, "onebody_twobody")
    assert (abs_dim, rel_dim) == (5, 5)
    g = out["test"][0]
    assert torch.allclose(g.onebody_features, (s["test"][0].ob.T - ob_mean) / ob_std)
    out, abs_dim, rel_dim = transform(s, "twobody_only")
    assert abs_dim == 0 and "onebody_features" not in out["train"][0]


@pytest.mark.parametrize("model", ["GCN", "GAT"])
@pytest.mark.parametrize("zinc_mode", ["zinc", "no_zinc"])
@pytest.mark.parametrize("config", ["no_pe", "onebody_twobody"])
def test_run_trial_smoke(model, zinc_mode, config):
    setting = "v2_gcn_shift" if model == "GCN" else "v2_gat_16h"
    r = run_trial(model, "ferro", config, "tiny", zinc_mode, setting, seed=0, max_epochs=2,
                  splits=splits())  # fmt: skip
    assert r["test_mae"] >= 0 and r["setting"] == setting


def _analyze():
    path = Path(__file__).parents[2] / "experiments/ablations/analyze.py"
    spec = importlib.util.spec_from_file_location("ablations_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analyze_summaries_reproduce_expected():
    a = _analyze()
    agg, per_size = a.summaries(pd.read_csv(a.EXPECTED / "trials.csv"))
    for got, name in ((agg, "summary_agg.csv"), (per_size, "summary_per_size.csv")):
        ref = pd.read_csv(a.EXPECTED / name)
        assert list(got.columns) == list(ref.columns) and len(got) == len(ref)
        assert np.allclose(got[["MAE_mean", "MAE_std"]], ref[["MAE_mean", "MAE_std"]], atol=1e-5)
        assert (got["count"] == ref["count"]).all()


def test_analyze_rerun_comparison(tmp_path):
    a = _analyze()
    ref = a.pooled_trials(None)
    paper = a.cells_frame(ref)
    same = a.compare(paper, paper)
    assert np.allclose(same.delta, 0) and "same best row in 2/2" in a.report(same)
    assert "quantum rows below it: 5/6" in a.claims(paper)
    # a re-run tree: every distinct trial of one Table 9 cell, original mean + 0.01
    sub = ref[(ref.model_type == "GCN") & (ref["size"] == "tiny") & (ref.config == "no_pe")]
    for t in sub.drop_duplicates(a.KEY).itertuples(index=False):
        d = tmp_path / "res" / f"{t.dataset}_{t.setting}_{t.seed}"
        d.mkdir(parents=True)
        r = {k: getattr(t, k) for k in a.KEY} | {k: 0 for k in a.TRIAL_COLS}
        r["test_mae"] = float(sub[(sub[a.KEY] == list(t[:7])).all(axis=1)].test_mae.mean()) + 0.01
        (d / "results.json").write_text(json.dumps(r))
    assert a.main(["--results", str(tmp_path / "res"), "--out", str(tmp_path / "out")]) == 0
    cmp = pd.read_csv(tmp_path / "out" / "rerun_comparison.csv")
    row = cmp[(cmp.table == "table9") & (cmp.row == "baseline") & (cmp.column == "tiny")]
    assert len(cmp) == 2 and abs(row.delta.item() - 0.01) < 1e-6  # Table 8 + Table 9 cell
    for name in ("rerun_trials.csv", "rerun_summary_agg.csv", "table8_ablation.tex"):
        assert (tmp_path / "out" / name).exists()
