"""Reference values of the paper (initial Annex H tables) and gap report.

Instantaneous references are matched on the stored time **index** (7, i.e. 1536 ns on the
data grid), not on the nearest time label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qgf.reconstruction.config import REFERENCE_TIME_INDEX

#: Tolerance of the reproduction criterion, in R^2.
DEFAULT_TOLERANCE: float = 0.02


@dataclass(frozen=True)
class ReferenceValue:
    """One row of the acceptance table.

    ``time_index=None`` means a time average over the grid (Table 13).
    """

    label: str
    dataset: str
    quench: str
    family: str
    probe: str
    k: int
    expected: float
    time_index: int | None = None
    tolerance: float = DEFAULT_TOLERANCE


def _ref(label, dataset, quench, family, probe, k, expected, instant=True) -> ReferenceValue:
    return ReferenceValue(
        label,
        dataset,
        quench,
        family,
        probe,
        k,
        expected,
        REFERENCE_TIME_INDEX if instant else None,
    )


#: Paper values. The nonlinear-probe references (Tables 14/15) were obtained with
#: row-level folds (information leak): graph folds reproduce the AFM rows but not FM.
REFERENCE_VALUES: tuple[ReferenceValue, ...] = (
    # Walk-power OLS, time averages (Table 13).
    _ref("OLS walk AFM MUTAG K=1", "MUTAG", "AFM", "walk", "ols", 1, 0.59, instant=False),
    _ref("OLS walk AFM MUTAG K=7", "MUTAG", "AFM", "walk", "ols", 7, 0.78, instant=False),
    _ref("OLS walk AFM MUTAG K=21", "MUTAG", "AFM", "walk", "ols", 21, 0.83, instant=False),
    _ref("OLS walk AFM MUTAG K=50", "MUTAG", "AFM", "walk", "ols", 50, 0.83, instant=False),
    _ref("OLS walk AFM ZINC K=1", "ZINC_small", "AFM", "walk", "ols", 1, 0.59, instant=False),
    _ref("OLS walk AFM ZINC K=7", "ZINC_small", "AFM", "walk", "ols", 7, 0.76, instant=False),
    _ref("OLS walk AFM ZINC K=21", "ZINC_small", "AFM", "walk", "ols", 21, 0.78, instant=False),
    _ref("OLS walk AFM ZINC K=50", "ZINC_small", "AFM", "walk", "ols", 50, 0.78, instant=False),
    # Nonlinear probes at the reference instant (Table 14).
    _ref("KR walk AFM MUTAG", "MUTAG", "AFM", "walk", "kernel_ridge", 21, 0.897),
    _ref("RF walk AFM MUTAG", "MUTAG", "AFM", "walk", "random_forest", 21, 0.910),
    _ref("KR rrwp AFM MUTAG", "MUTAG", "AFM", "rrwp", "kernel_ridge", 21, 0.898),
    _ref("RF rrwp AFM MUTAG", "MUTAG", "AFM", "rrwp", "random_forest", 21, 0.912),
    _ref("KR walk FM MUTAG", "MUTAG", "FM", "walk", "kernel_ridge", 21, 0.765),
    _ref("RF walk FM MUTAG", "MUTAG", "FM", "walk", "random_forest", 21, 0.805),
    _ref("KR rrwp FM MUTAG", "MUTAG", "FM", "rrwp", "kernel_ridge", 21, 0.704),
    _ref("RF rrwp FM MUTAG", "MUTAG", "FM", "rrwp", "random_forest", 21, 0.776),
    # Laplacian augmentation (Table 15).
    _ref("RF walk+lap AFM MUTAG", "MUTAG", "AFM", "walk+laplacian", "random_forest", 21, 0.915),
    _ref("RF rrwp+lap AFM MUTAG", "MUTAG", "AFM", "rrwp+laplacian", "random_forest", 21, 0.916),
    _ref("RF walk+lap FM MUTAG", "MUTAG", "FM", "walk+laplacian", "random_forest", 21, 0.915),
    _ref("RF rrwp+lap FM MUTAG", "MUTAG", "FM", "rrwp+laplacian", "random_forest", 21, 0.918),
)

#: Expected per-graph statistics at K = n_alpha - 1 (Cayley-Hamilton).
PER_GRAPH_REFERENCE: dict[str, float] = {
    "walk_median": 0.88,
    "walk_iqr_low": 0.85,
    "walk_iqr_high": 0.90,
    "rrwp_median": 0.885,
}


def gap_report(
    results: pd.DataFrame,
    values: tuple[ReferenceValue, ...] = REFERENCE_VALUES,
) -> pd.DataFrame:
    """Row-by-row gap report against the reference values (missing rows give NaN)."""
    rows = []
    for value in values:
        selected = results[
            (results["dataset"] == value.dataset)
            & (results["quench"] == value.quench)
            & (results["family"] == value.family)
            & (results["probe"] == value.probe)
            & (results["k"] == value.k)
        ]
        if value.time_index is None:
            aggregation = "time average"
        else:
            selected = selected[selected["time_index"] == value.time_index]
            aggregation = "instant"
        observed = float(selected["r2"].mean()) if not selected.empty else np.nan
        gap = observed - value.expected
        rows.append(
            {
                "label": value.label,
                "dataset": value.dataset,
                "quench": value.quench,
                "family": value.family,
                "probe": value.probe,
                "k": value.k,
                "aggregation": aggregation,
                "time_index": value.time_index,
                "expected": value.expected,
                "observed": observed,
                "gap": gap,
                "within_tolerance": bool(np.isfinite(gap) and abs(gap) <= value.tolerance),
            },
        )
    return pd.DataFrame(rows)


def coherence_report(pooled: pd.DataFrame, per_graph: pd.DataFrame) -> pd.DataFrame:
    """Pooled R^2 vs per-graph mean R^2_pg = N^-1 sum_alpha R^2_alpha."""
    per_graph_mean = (
        per_graph.groupby(["dataset", "quench", "family", "time_index"])["r2"]
        .mean()
        .rename("r2_per_graph")
        .reset_index()
    )
    pooled_max_k = (
        pooled[pooled["probe"] == "ols"]
        .sort_values("k")
        .groupby(["dataset", "quench", "family", "time_index"])
        .tail(1)[["dataset", "quench", "family", "time_index", "k", "r2"]]
        .rename(columns={"r2": "r2_pooled", "k": "k_pooled"})
    )
    merged = per_graph_mean.merge(
        pooled_max_k,
        on=["dataset", "quench", "family", "time_index"],
        how="inner",
    )
    merged["gap"] = merged["r2_pooled"] - merged["r2_per_graph"]
    return merged


def per_graph_summary(per_graph: pd.DataFrame) -> pd.DataFrame:
    """Median, IQR and worst case of the per-graph R^2, per family and quench."""
    grouped = per_graph.groupby(["dataset", "quench", "family", "graph_id"])["r2"].mean()
    frame = grouped.rename("r2").reset_index()
    summary = (
        frame.groupby(["dataset", "quench", "family"])["r2"]
        .agg(
            median="median",
            iqr_low=lambda values: float(np.percentile(values, 25)),
            iqr_high=lambda values: float(np.percentile(values, 75)),
            worst="min",
        )
        .reset_index()
    )
    worst_ids = frame.loc[frame.groupby(["dataset", "quench", "family"])["r2"].idxmin()]
    return summary.merge(
        worst_ids[["dataset", "quench", "family", "graph_id"]].rename(
            columns={"graph_id": "worst_graph_id"},
        ),
        on=["dataset", "quench", "family"],
    )
