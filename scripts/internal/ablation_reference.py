"""Collect the per-trial results of the original ablation runs (Tables 8-10).

Internal use only: reads the aggregated ``all_results.csv`` of the original study (path
given as argument); not needed to reproduce results from the released data.

Each original run directory is mapped to its training setting
(``qgf.downstream.gnns.ablation.SETTINGS``); the per-trial table keeps every row, including
the seeds that several sweeps re-ran (the paper's cells pool them as they are).

    python scripts/internal/ablation_reference.py --all-results <all_results.csv> \
        --out experiments/ablations/expected
"""

import argparse
from pathlib import Path

import pandas as pd

RUN_SETTING = {
    "20260331_030951_GCN": "v1",
    "20260331_032414_GCN": "v1",
    "20260331_175904_GCN_large_models": "v1",
    "20260401_140818_GCN_all_sizes": "v1",
    "20260401_213304_GCN_rerun_fixed": "v1",
    "20260402_163356_GCN_v2": "v2_gcn_shift",
    "20260402_164007_GCN_v2": "v2_gcn_shift",
    "20260403_130715_GCN_clamp": "v2_gcn_clamp",
    "20260331_030956_GAT": "v1",
    "20260331_032415_GAT": "v1",
    "20260331_175910_GAT_large_models": "v1",
    "20260401_140824_GAT_all_sizes": "v1",
    "20260401_213313_GAT_rerun_fixed": "v1",
    "20260402_164015_GAT_v2": "v2_gat",
    "20260402_164647_GAT_16h": "v2_gat_16h",
}
KEYS = ["model_type", "dataset", "config", "zinc_mode"]


def summaries(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``summary_per_size`` and ``summary_agg`` (mean, std ddof 1, count) as the original."""
    agg = {"MAE_mean": ("test_mae", "mean"), "MAE_std": ("test_mae", "std"),
           "count": ("test_mae", "size")}  # fmt: skip
    per_size = trials.groupby([*KEYS, "size"]).agg(**agg).reset_index()
    pooled = trials.groupby(KEYS).agg(**agg).reset_index()
    return per_size, pooled


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    d = pd.read_csv(args.all_results)
    d["run"] = d.path.str.split("/").str[-7]
    d["setting"] = d.run.map(RUN_SETTING)
    if d.setting.isna().any():
        raise ValueError(f"unmapped runs: {sorted(d.run[d.setting.isna()].unique())}")
    v2 = d[d.setting != "v1"]  # the v2 runs recorded their flags: check the mapping
    heads = v2.setting.map({"v2_gat_16h": 16}).fillna(1)
    assert (v2.num_heads.fillna(1) == heads).all(), "num_heads mismatch"
    shift = v2.setting == "v2_gcn_shift"
    assert (v2.enable_edge_weight_shift.astype(bool) == shift).all(), "shift mismatch"
    cols = ["model_type", "dataset", "config", "size", "zinc_mode", "setting", "seed", "run",
            "test_mae", "best_epoch", "num_params"]  # fmt: skip
    trials = d[cols].sort_values(cols[:8])
    args.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out / "trials.csv", index=False, float_format="%.6f")
    per_size, pooled = summaries(trials)
    per_size.to_csv(args.out / "summary_per_size.csv", index=False, float_format="%.6f")
    pooled.to_csv(args.out / "summary_agg.csv", index=False, float_format="%.6f")
    unique = trials.drop_duplicates(["model_type", "dataset", "config", "size", "zinc_mode",
                                     "setting", "seed"])  # fmt: skip
    print(f"{len(trials)} rows, {len(unique)} unique (cell, setting, seed) trials")


if __name__ == "__main__":
    main()
