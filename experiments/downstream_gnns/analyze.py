"""Aggregate downstream GNN trials into the paper tables.

Cells (Tables 2-3, geometry table): mean and standard deviation (ddof 1) over all trials
of a (backbone, config): 10 folds x 5 seeds on MUTAG (macro-F1), 5 seeds on ZINC (MAE).
The paper's MUTAG hardware rows report instead the std (ddof 1) of the 10 fold means
(column ``std_fold``).

    uv run python experiments/downstream_gnns/analyze.py --results results/downstream_gnns \
        [--expected experiments/downstream_gnns/expected]

Writes ``<results>/<dataset>_summary.csv`` and ``<results>/<dataset>_trials.csv`` (one row
per trial, columns of ``expected/<dataset>_trials.csv``); with ``--expected`` also the per-cell
difference to the reference runs in units of the reference standard deviation and of the
two-sample standard error (``<results>/<dataset>_comparison.csv``), the orderings and the
claims (``<results>/<dataset>_claims.csv``, reference and re-run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRIC = {"mutag": "test_f1_macro", "zinc": "test_mae"}


def read_tree(root: Path, dataset: str) -> pd.DataFrame:
    """One row per trial: model, config, fold (-1 on ZINC), seed, value, encoding dims."""
    rows = []
    for p in sorted((root / dataset).glob("*/*/**/results.json")):
        r = json.loads(p.read_text())
        fold = -1 if r["fold"] is None else r["fold"]
        rows.append(
            (
                r["model_type"],
                r["config"],
                fold,
                r["seed"],
                r[METRIC[dataset]],
                r.get("abs_pe_dim"),
                r.get("rel_pe_dim"),
            )
        )
    cols = ["model", "config", "fold", "seed", "value", "abs_pe_dim", "rel_pe_dim"]
    return pd.DataFrame(rows, columns=cols)


def aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (model, config): mean, std (ddof 1) over trials, std (ddof 1) over fold means
    (MUTAG only; the convention of the paper's hardware rows), number of trials."""
    keys = ["model", "config"]
    out = trials.groupby(keys, sort=False)["value"].agg(mean="mean", std="std", n_trials="size")
    fold_means = trials[trials.fold >= 0].groupby([*keys, "fold"])["value"].mean()
    out["std_fold"] = fold_means.groupby(level=keys).std()
    return out.reset_index()


def compare(summary: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Per cell present in both: ``|mean - mean_ref| / std_ref`` and, when the reference
    gives its number of trials, the difference in units of its two-sample standard error
    ``sqrt(std^2 / n + std_ref^2 / n_ref)`` (the relevant scale with 5 ZINC seeds)."""
    m = summary.merge(expected, on=["model", "config"], suffixes=("", "_ref"))
    m["abs_delta"] = (m["mean"] - m["mean_ref"]).abs()
    m["delta_over_std"] = m["abs_delta"] / m["std_ref"]
    cols = ["model", "config", "mean", "std", "mean_ref", "std_ref", "abs_delta"]
    cols += ["delta_over_std", "n_trials"]
    if "n_trials_ref" in m:
        se = (m["std"] ** 2 / m["n_trials"] + m["std_ref"] ** 2 / m["n_trials_ref"]) ** 0.5
        m["delta_over_se"] = m["abs_delta"] / se
        cols += ["n_trials_ref", "delta_over_se"]
    return m[cols]


def ranking(
    summary: pd.DataFrame,
    expected: pd.DataFrame,
    dataset: str,
    configs: list[str] | None = None,
) -> pd.DataFrame:
    """Per backbone, over the feature sets present in both (restricted to ``configs`` if
    given, e.g. the table rows): best and runner-up (the paper's markings) of the re-run and
    of the reference, and Kendall's tau of the means."""
    from scipy.stats import kendalltau

    better_high = dataset == "mutag"  # F1 up, MAE down
    rows = []
    m = summary.merge(expected, on=["model", "config"], suffixes=("", "_ref"))
    if configs is not None:
        m = m[m.config.isin(configs)]
    for model, g in m.groupby("model"):
        mine = g.sort_values("mean", ascending=not better_high).config.tolist()
        ref = g.sort_values("mean_ref", ascending=not better_high).config.tolist()
        rows.append(
            {
                "model": model,
                "n_configs": len(g),
                "best": mine[0],
                "best_ref": ref[0],
                "second": mine[1] if len(mine) > 1 else "",
                "second_ref": ref[1] if len(ref) > 1 else "",
                "kendall_tau": kendalltau(g["mean"], g["mean_ref"]).statistic,
            }
        )
    return pd.DataFrame(rows)


CLASSICAL = ["no_pe", "laplacian", "rrwp", "rrwp_laplacian"]
TABLE2 = CLASSICAL + [
    "af",
    "af_stat",
    "fm",
    "fm_stat",
    "af_fm",
    "rrwp_af",
    "rrwp_fm",
    "rrwp_af_stat",
    "rrwp_fm_stat",
    "rrwp_af_fm",
    "rrwp_af_stat_fm_stat",
]
HARDWARE = ["hw_afm", "hw_afm_stat", "rrwp_hw_afm", "rrwp_hw_afm_stat"]
TABLE_ROWS = {"mutag": TABLE2 + HARDWARE, "zinc": TABLE2}  # rows of Tables 2 and 3


def claims(summary: pd.DataFrame, dataset: str = "mutag") -> pd.DataFrame:
    """Claims per backbone: Table 2 / 3 column best (an RRWP + quantum hybrid?), fixed
    champion gap (RRWP + AFM_stat + FM_stat minus RRWP), geometry-table difference of
    + AFM + FM, and on MUTAG the hardware gaps (RRWP + HW-AFM[_stat] minus RRWP + AFM[_stat]).
    Gaps are raw differences of the metric (MUTAG F1: > 0 favours the first set; ZINC MAE:
    < 0 favours it)."""
    s = summary.set_index(["model", "config"])["mean"]
    pick = max if dataset == "mutag" else min
    rows = []
    for model in sorted({m for m, _ in s.index}):
        t2 = {c: s[(model, c)] for c in TABLE2 if (model, c) in s.index}
        best = pick(t2, key=t2.get) if t2 else ""

        def gap(a: str, b: str, m: str = model) -> float:
            return s.get((m, a), float("nan")) - s.get((m, b), float("nan"))

        row = {
            "model": model,
            "table2_best" if dataset == "mutag" else "table3_best": best,
            "best_is_hybrid": best.startswith("rrwp_") and best != "rrwp_laplacian",
            "champion_gap": gap("rrwp_af_stat_fm_stat", "rrwp"),
            "geometry_gain": gap("rrwp_lap_ring_geom_af_fm", "rrwp_lap_ring_geom"),
        }
        if dataset == "mutag":
            row["hw_stat_gap"] = gap("rrwp_hw_afm_stat", "rrwp_af_stat")
            row["hw_full_gap"] = gap("rrwp_hw_afm", "rrwp_af")
        rows.append(row)
    return pd.DataFrame(rows)


def pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Config x backbone table of ``mean ± std`` strings."""
    cells = summary.assign(
        cell=summary["mean"].map("{:.3f}".format) + " ± " + summary["std"].map("{:.3f}".format)
    )
    return cells.pivot(index="config", columns="model", values="cell")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results/downstream_gnns"))
    p.add_argument("--datasets", nargs="+", default=["mutag", "zinc"])
    p.add_argument("--expected", type=Path, default=None)
    args = p.parse_args(argv)
    pd.set_option("display.width", 220)
    for ds in args.datasets:
        trials = read_tree(args.results, ds)
        if trials.empty:
            print(f"[{ds}] no results under {args.results / ds}")
            continue
        summary = aggregate(trials)
        summary.to_csv(args.results / f"{ds}_summary.csv", index=False, float_format="%.6f")
        trials.assign(source="rerun").to_csv(
            args.results / f"{ds}_trials.csv", index=False, float_format="%.6f"
        )
        print(f"\n[{ds}] {METRIC[ds]}, mean ± std (ddof 1) over trials")
        print(pivot(summary).to_string())
        if args.expected is not None:
            ref = pd.read_csv(args.expected / f"{ds}_summary.csv")
            diff = compare(summary, ref)
            diff.to_csv(args.results / f"{ds}_comparison.csv", index=False, float_format="%.6f")
            print(diff.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print(
                f"[{ds}] |delta|/std max {diff['delta_over_std'].max():.2f}, "
                f"mean {diff['delta_over_std'].mean():.2f} over {len(diff)} cells"
            )
            both = pd.concat(
                [claims(ref, ds).assign(src="reference"), claims(summary, ds).assign(src="rerun")]
            )
            both.to_csv(args.results / f"{ds}_claims.csv", index=False, float_format="%.4f")
            print(both.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            print(
                ranking(summary, ref, ds, TABLE_ROWS[ds]).to_string(
                    index=False, float_format=lambda v: f"{v:.2f}"
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
