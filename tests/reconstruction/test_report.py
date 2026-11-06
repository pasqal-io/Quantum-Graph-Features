"""Deliverables: serialisation, tables, extension protocol, gap report."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from qgf.reconstruction.reference import (
    REFERENCE_VALUES,
    coherence_report,
    gap_report,
    per_graph_summary,
)
from qgf.reconstruction.report import (
    extension_report,
    plot_heatmaps,
    plot_residual_curves,
    quench_contrast,
    read_results,
    table_at_time,
    table_time_averaged,
    to_latex,
    write_results,
)


def _results_frame() -> pd.DataFrame:
    rows = []
    values = {
        ("walk", "AFM"): 0.80,
        ("walk", "FM"): 0.70,
        ("geometry", "AFM"): 0.40,
        ("geometry", "FM"): 0.35,
        ("walk+geometry", "AFM"): 0.85,
        ("walk+geometry", "FM"): 0.82,
    }
    columns = {"walk": 22, "geometry": 27, "walk+geometry": 49}
    for (family, quench), value in values.items():
        for k in (1, 21):
            for step, time_ns in enumerate((2273.0, 4000.0)):
                rows.append(
                    {
                        "dataset": "MUTAG",
                        "quench": quench,
                        "family": family,
                        "phase": "P1" if family == "walk" else "P2",
                        "k": k,
                        "probe": "ols",
                        "time_index": step,
                        "time_ns": time_ns,
                        "r2": value if k == 21 else value - 0.2,
                        "r2_adjusted": value - 0.01,
                        "num_columns": columns[family],
                        "num_parameters": columns[family],
                        "num_rows": 30505,
                        "fold_mode": "in_sample",
                        "intercept": False,
                        "seed": 0,
                    },
                )
    return pd.DataFrame(rows)


def test_write_results_round_trips(tmp_path):
    """Parquet round trip keeps the whole index."""
    frame = _results_frame()
    path = write_results(frame, str(tmp_path / "results.parquet"))
    assert os.path.exists(path)
    reloaded = read_results(path)
    assert len(reloaded) == len(frame)
    assert set(reloaded["family"]) == set(frame["family"])


def test_table_time_averaged_is_indexed_by_k():
    table = table_time_averaged(_results_frame())
    assert list(table.index) == [1, 21]
    assert ("MUTAG", "AFM", "walk") in table.columns
    assert table.loc[21, ("MUTAG", "AFM", "walk")] == pytest.approx(0.80)


def test_table_at_time_selects_the_time_index():
    table = table_at_time(_results_frame(), quench="AFM", time_index=0)
    assert table.attrs["time_ns"] == 2273.0
    assert ("MUTAG", "walk", 21) in table.index


def test_table_at_time_is_empty_for_missing_quench():
    assert table_at_time(_results_frame(), quench="XX").empty


def test_to_latex_emits_a_tabular():
    latex = to_latex(table_time_averaged(_results_frame()), "caption", "tab:test")
    assert "tabular" in latex
    assert "tab:test" in latex


def test_extension_report_computes_incremental_and_residual_gain():
    report = extension_report(_results_frame(), baselines=("walk",))
    row = report[
        (report["extension"] == "geometry")
        & (report["quench"] == "AFM")
        & (report["k"] == 21)
        & (report["time_ns"] == 2273.0)
    ].iloc[0]
    assert row["r2_baseline"] == pytest.approx(0.80)
    assert row["r2_solo"] == pytest.approx(0.40)
    assert row["delta_r2"] == pytest.approx(0.05)
    # 0.05 / (1 - 0.80) = 25 % of the residual closed.
    assert row["residual_fraction"] == pytest.approx(0.25)
    assert row["extra_columns"] == 27


def test_extension_report_is_empty_without_combinations():
    frame = _results_frame()
    solo_only = frame[~frame["family"].str.contains(r"\+")]
    assert extension_report(solo_only).empty


def test_quench_contrast_reports_both_residuals():
    contrast = quench_contrast(_results_frame())
    row = contrast[
        (contrast["family"] == "walk") & (contrast["k"] == 21) & (contrast["time_ns"] == 2273.0)
    ].iloc[0]
    assert row["afm_minus_fm"] == pytest.approx(0.10)
    assert row["residual_afm"] == pytest.approx(0.20)
    assert row["residual_fm"] == pytest.approx(0.30)


def test_gap_report_covers_every_reference_line():
    report = gap_report(_results_frame())
    assert len(report) == len(REFERENCE_VALUES)
    # Rows not covered by the run are reported, not ignored.
    assert report["observed"].isna().any()
    assert not report.loc[report["observed"].isna(), "within_tolerance"].any()


def test_gap_report_flags_within_tolerance():
    frame = _results_frame()
    frame.loc[
        (frame["family"] == "walk") & (frame["quench"] == "AFM") & (frame["k"] == 21),
        "r2",
    ] = 0.83
    report = gap_report(frame)
    line = report[report["label"] == "OLS walk AFM MUTAG K=21"].iloc[0]
    assert line["observed"] == pytest.approx(0.83)
    assert line["gap"] == pytest.approx(0.0)
    assert bool(line["within_tolerance"])


def _per_graph_frame() -> pd.DataFrame:
    rows = []
    for graph_id in range(6):
        for time_ns in (2273.0, 4000.0):
            rows.append(
                {
                    "dataset": "MUTAG",
                    "quench": "AFM",
                    "family": "walk",
                    "graph_id": graph_id,
                    "num_nodes": 10 + graph_id,
                    "k": 9 + graph_id,
                    "time_index": 0 if time_ns == 2273.0 else 1,
                    "time_ns": time_ns,
                    "r2": 0.5 + 0.08 * graph_id,
                },
            )
    return pd.DataFrame(rows)


def test_per_graph_summary_reports_median_iqr_and_worst():
    summary = per_graph_summary(_per_graph_frame()).iloc[0]
    assert summary["worst"] == pytest.approx(0.5)
    assert summary["worst_graph_id"] == 0
    assert summary["iqr_low"] < summary["median"] < summary["iqr_high"]


def test_coherence_report_compares_pooled_and_per_graph():
    coherence = coherence_report(_results_frame(), _per_graph_frame())
    assert not coherence.empty
    row = coherence[coherence["time_index"] == 0].iloc[0]
    assert row["k_pooled"] == 21
    assert row["gap"] == pytest.approx(row["r2_pooled"] - row["r2_per_graph"])


def test_heatmap_and_residual_figures_are_written(tmp_path):
    frame = _results_frame()
    heatmap = plot_heatmaps(frame, str(tmp_path / "figures" / "heatmap.pdf"))
    curves = plot_residual_curves(frame, str(tmp_path / "figures" / "residual.pdf"))
    assert os.path.getsize(heatmap) > 0
    assert os.path.getsize(curves) > 0


def test_heatmap_rejects_unknown_probe(tmp_path):
    with pytest.raises(ValueError, match="No result"):
        plot_heatmaps(_results_frame(), str(tmp_path / "x.pdf"), probe="kernel_ridge")


def test_extension_report_handles_nan_residual():
    """A baseline R² of 1.0 makes the relative residual gain undefined, not infinite."""
    frame = _results_frame()
    frame.loc[frame["family"] == "walk", "r2"] = 1.0
    report = extension_report(frame, baselines=("walk",))
    assert np.all(np.isnan(report["residual_fraction"]))


def test_extension_report_attributes_gain_to_the_richest_baseline():
    """A gain must be measured above both walk and walk+laplacian."""
    rows = []
    values = {
        "walk": 0.80,
        "walk+laplacian": 0.85,
        "interaction": 0.50,
        "walk+interaction": 0.88,
        "walk+laplacian+interaction": 0.90,
    }
    for family, value in values.items():
        rows.append(
            {
                "dataset": "MUTAG",
                "quench": "FM",
                "family": family,
                "phase": "P2",
                "k": 21,
                "probe": "ols",
                "time_index": 0,
                "time_ns": 2273.0,
                "r2": value,
                "r2_adjusted": value - 0.01,
                "num_columns": 22,
                "num_parameters": 22,
                "num_rows": 30505,
                "fold_mode": "in_sample",
                "intercept": True,
                "seed": 0,
            },
        )
    report = extension_report(pd.DataFrame(rows))
    over_walk = report[(report["baseline"] == "walk") & (report["extension"] == "interaction")]
    over_spectral = report[report["baseline"] == "walk+laplacian"]
    assert over_walk["delta_r2"].iloc[0] == pytest.approx(0.08)
    assert over_spectral["extension"].iloc[0] == "interaction"
    # 0.90 - 0.85 = 0.05, i.e. one third of the walk+laplacian residual.
    assert over_spectral["delta_r2"].iloc[0] == pytest.approx(0.05)
    assert over_spectral["residual_fraction"].iloc[0] == pytest.approx(0.05 / 0.15)


def test_classical_ceiling_is_locked_and_excludes_diagnostics():
    """The classical ceiling contains only the atoms that survive the forest."""
    from qgf.reconstruction.registry import (
        CLASSICAL_CEILING,
        CLASSICAL_CEILING_ATOMS,
        CLASSICAL_DIAGNOSTICS,
    )

    assert "+".join(CLASSICAL_CEILING_ATOMS) == CLASSICAL_CEILING
    # jlaplacian degraded the FM, jpow/jrrwp are redundant, chemical is a control.
    for excluded in ("jlaplacian", "jpow", "jrrwp", "chemical", "geometry"):
        assert excluded not in CLASSICAL_CEILING_ATOMS
        assert excluded in CLASSICAL_DIAGNOSTICS
    # Les briques structurelles retenues y sont toutes.
    for kept in ("walk", "rrwp", "laplacian", "ring", "interaction", "distance"):
        assert kept in CLASSICAL_CEILING_ATOMS
