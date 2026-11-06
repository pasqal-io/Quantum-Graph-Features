"""Collect the per-trial results of the original GNN runs into the expected tables.

Internal use only: reads the result tree of the original runs (path given as argument);
not needed to reproduce results from the released data.

Sources per cell (backbones GCN, GAT, GINE, Transformer, General; size ``large``):

- Tables 2-3, all feature sets except the two Laplacian rows: the v2 multi-model runs
  (``all_{mutag,zinc}_results.csv``);
- ``rrwp_laplacian``: the v3 re-run with the full 4-eigenvector Laplacian (dim 24; v2 had
  21), as in the paper;
- ``laplacian``: the v3 Laplacian-only re-run (dim 4). The paper's Laplacian row comes from
  the v2 run, where the Laplacian block had dimension 1; the port computes 4 eigenvectors,
  so the v3 run is its reference (the v2 values are kept in ``paper_laplacian_v2``);
- HW rows: the 4-time-step QPU run ``*_afm_qpu_large_v2_4steps`` (configs ``afm_qpu*``
  renamed ``hw_afm*``);
- geometry table: the strong-baseline v4 runs (``rrwp_laplacian_ring_geom*`` renamed
  ``rrwp_lap_ring_geom*``; their ring and geometry encodings differ from the GT runs).

    python scripts/internal/downstream_gnns_reference.py --results <qet>/results \
        --out experiments/downstream_gnns/expected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = ["GCN", "GAT", "GINE", "Transformer", "General"]
RENAME = {
    "afm_qpu": "hw_afm",
    "afm_qpu_stat": "hw_afm_stat",
    "rrwp_afm_qpu": "rrwp_hw_afm",
    "rrwp_afm_qpu_stat": "rrwp_hw_afm_stat",
    "rrwp_laplacian_ring_geom": "rrwp_lap_ring_geom",
    "rrwp_laplacian_ring_geom_af_fm": "rrwp_lap_ring_geom_af_fm",
}
METRIC = {"mutag": "test_f1_macro", "zinc": "test_mae"}
QPU_RUN = "PE_template_MUTAG/20260430_155033_multimodel_cpu_afm_qpu_large_v2_4steps"


def from_csv(path: Path, dataset: str, source: str, configs=None, drop=()) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.model_type.isin(MODELS) & ~df.config.isin(drop)]
    if configs is not None:
        df = df[df.config.isin(configs)]
    fold = df.get("fold", -1)
    return pd.DataFrame(
        {
            "model": df.model_type,
            "config": df.config.replace(RENAME),
            "fold": fold,
            "seed": df.seed,
            "value": df[METRIC[dataset]],
            "abs_pe_dim": df.abs_pe_dim,
            "rel_pe_dim": df.rel_pe_dim,
            "source": source,
        }
    )


def from_tree(root: Path, source: str) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/*/fold_*/seed_*/results.json")):
        r = json.loads(p.read_text())
        if r["model_type"] in MODELS:
            rows.append(
                (
                    r["model_type"],
                    RENAME.get(r["config"], r["config"]),
                    r["fold"],
                    r["seed"],
                    r["test_f1_macro"],
                    r["abs_pe_dim"],
                    r["rel_pe_dim"],
                    source,
                )
            )
    cols = ["model", "config", "fold", "seed", "value", "abs_pe_dim", "rel_pe_dim", "source"]
    return pd.DataFrame(rows, columns=cols)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby(["model", "config"], sort=False)
    out = g["value"].agg(mean="mean", std="std", n_trials="size")
    # MUTAG: std of the 10 fold means (ddof 1), the convention of the paper's HW rows
    out["std_fold"] = (
        trials.groupby(["model", "config", "fold"])
        .value.mean()
        .groupby(level=["model", "config"])
        .std()
    )
    out.loc[trials.groupby(["model", "config"]).fold.max() < 0, "std_fold"] = float("nan")
    out["abs_pe_dim"] = g["abs_pe_dim"].first()
    out["rel_pe_dim"] = g["rel_pe_dim"].first()
    out["source"] = g["source"].first()
    return out.reset_index()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True, help="<qet>/results")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    v2 = args.results / "final_results_for_gnns_table" / "analysis_multimodel_large_v2"
    v3 = args.results / "experiences" / "analysis_multimodel_large_v2"
    v4 = args.results / "final_results_for_gnns_table" / "analysis_strong_baseline"
    lap = ["laplacian", "rrwp_laplacian"]
    geo = list(RENAME)[4:]
    parts = {
        "mutag": [
            from_csv(v2 / "all_mutag_results.csv", "mutag", "v2", drop=lap),
            from_csv(v3 / "v3_mutag_full_laplacian.csv", "mutag", "v3_full_laplacian"),
            from_csv(v3 / "v3_mutag_laplacian_only.csv", "mutag", "v3_laplacian_only"),
            from_tree(args.results / "experiences" / QPU_RUN, "qpu_4steps"),
            from_csv(v4 / "v4_mutag_strong_baseline.csv", "mutag", "v4_strong", configs=geo),
        ],
        "zinc": [
            from_csv(v2 / "all_zinc_results.csv", "zinc", "v2", drop=lap),
            from_csv(v3 / "v3_zinc_full_laplacian.csv", "zinc", "v3_full_laplacian"),
            from_csv(v3 / "v3_zinc_laplacian_only.csv", "zinc", "v3_laplacian_only"),
            from_csv(v4 / "v4_zinc_strong_baseline.csv", "zinc", "v4_strong", configs=geo),
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for ds, frames in parts.items():
        trials = pd.concat(frames, ignore_index=True)
        trials = trials[~trials.config.str.startswith("gs_")]  # ground-state pulses: not in paper
        dup = trials.duplicated(["model", "config", "fold", "seed"])
        if dup.any():
            raise ValueError(f"{ds}: duplicate trials\n{trials[dup]}")
        trials.to_csv(args.out / f"{ds}_trials.csv", index=False, float_format="%.6f")
        summarize(trials).to_csv(args.out / f"{ds}_summary.csv", index=False, float_format="%.6f")
        paper_lap = from_csv(v2 / f"all_{ds}_results.csv", ds, "v2", configs=["laplacian"])
        summarize(paper_lap).to_csv(
            args.out / f"{ds}_paper_laplacian_v2.csv", index=False, float_format="%.6f"
        )
        print(f"{ds}: {len(trials)} trials, {trials.groupby(['model', 'config']).ngroups} cells")


if __name__ == "__main__":
    main()
