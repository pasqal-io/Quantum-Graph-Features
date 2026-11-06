"""Logic tests of the App. G statistics port (synthetic trials, no Stan)."""

import json

import numpy as np
import pytest

from qgf.stats import pe_benchmark as pb


def synthetic_trials(gain: float = 0.01) -> pb.TrialTable:
    """All 6 archs x MUTAG (+ GT ZINC) x 16 configs; rrwp_afm_stat_fm_stat best hybrid."""
    rng = np.random.default_rng(0)
    trials: pb.TrialTable = {}
    for arch in pb.ARCHITECTURES:
        for cfg in pb.CANONICAL_CONFIGS:
            base = 0.85 if cfg == "rrwp" else 0.7
            if cfg == pb.HARDCODED_QUANTUM_CHAMPION:
                base = 0.85 + gain
            trials[(arch, "MUTAG", cfg)] = {
                (f, s): base + rng.normal(0, 0.05) for f in range(10) for s in range(1, 6)
            }
            if arch == "GT":
                zbase = (
                    0.114 if cfg == "rrwp" else (0.106 if cfg.startswith("rrwp_afm_stat") else 0.2)
                )
                trials[(arch, "ZINC_small", cfg)] = {
                    (-1, s): zbase + rng.normal(0, 0.005) for s in range(10)
                }
    return trials


def test_champions_and_sign_test():
    df = pb.build_cell_aggregates(pb.build_all_trials(synthetic_trials()))
    assert len(df) == 6 * 16 + 16 + 5 * 16  # MUTAG cells + GT ZINC + ZINC summaries
    classical, quantum = pb.assert_champions(df)
    r = pb.sign_test_5b(df, classical, quantum)
    assert r["n_cells"] == 12
    from scipy.stats import binomtest

    assert r["p_value"] == pytest.approx(binomtest(r["k_wins"], 12, 0.5, "greater").pvalue)


def test_signed_diff_convention():
    assert pb.signed_diff(0.9, 0.8, "MUTAG") == pytest.approx(0.1)
    assert pb.signed_diff(0.10, 0.12, "ZINC_small") == pytest.approx(0.02)  # lower MAE wins


def test_paper_zinc_summary_gives_10_of_12():
    """With the real Table 3 summaries and GT gains on both datasets, 10/12, p = 0.019."""
    df = pb.build_cell_aggregates(pb.build_all_trials(synthetic_trials()))
    r = pb.sign_test_5b(df, "rrwp", pb.HARDCODED_QUANTUM_CHAMPION)
    losses = {(c["arch"], c["dataset"]) for c in r["per_cell"] if c["diff"] <= 0}
    assert {("Transformer", "ZINC_small"), ("GeneralConv", "ZINC_small")} <= losses


def test_csv_and_tree_readers(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "arch,dataset,config,fold,seed,value\nGCN,MUTAG,rrwp,0,1,0.8\nGCN,MUTAG,rrwp,0,2,0.9\n"
    )
    t = pb.read_trials_csv(csv_path)
    assert t[("GCN", "MUTAG", "rrwp")] == {(0, 1): 0.8, (0, 2): 0.9}
    for fold, seed, v in [(0, 1, 0.7), (1, 1, 0.6)]:
        d = tmp_path / "gt/af_stat" / f"fold_{fold}" / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({"test_f1_macro": v}))
    g = pb.read_gt_results_tree(tmp_path / "gt", "MUTAG")
    assert g[("GT", "MUTAG", "afm_stat")] == {(0, 1): 0.7, (1, 1): 0.6}


def test_t_posterior_interval_contains_mean():
    d = np.random.default_rng(1).normal(0.01, 0.02, 300)
    r = pb._t_posterior(d, seed=42)
    assert r["hdi_lo"] < d.mean() < r["hdi_hi"]
    assert r["median"] == pytest.approx(d.mean(), abs=5e-4)


def test_read_gnn_results_tree(tmp_path):
    for fold, seed, value in ((0, 1, 0.8), (1, 2, 0.9)):
        d = tmp_path / "General" / "rrwp_af_stat_fm_stat" / f"fold_{fold}" / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(
            json.dumps({"fold": fold, "seed": seed, "test_f1_macro": value})
        )
    table = pb.read_gnn_results_tree(tmp_path)
    assert table == {("GeneralConv", "MUTAG", "rrwp_afm_stat_fm_stat"): {(0, 1): 0.8, (1, 2): 0.9}}


def test_gnn_trials_csv(tmp_path):
    import importlib.util
    from pathlib import Path

    d = tmp_path / "GCN" / "rrwp" / "fold_3" / "seed_2"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"fold": 3, "seed": 2, "test_f1_macro": 0.7}))
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "gnn_trials", root / "experiments/stats/gnn_trials.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main([str(tmp_path), str(tmp_path / "t.csv")])
    assert pb.read_trials_csv(tmp_path / "t.csv") == {("GCN", "MUTAG", "rrwp"): {(3, 2): 0.7}}
