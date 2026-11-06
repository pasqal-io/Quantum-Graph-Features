"""Smoke tests of the experiment entry points on a tiny MUTAG subset (needs the data)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qgf.data import DEFAULT_ROOT
from qgf.reconstruction import run as recon_run
from qgf.reconstruction.residual_dataset import compute_residuals

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow
needs_data = pytest.mark.skipif(
    not (Path(DEFAULT_ROOT) / "mutag_afm.pt").exists(),
    reason="released datasets not available",
)


def _experiment():
    spec = importlib.util.spec_from_file_location(
        "recon_experiment",
        ROOT / "experiments" / "reconstruction" / "run.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_data
def test_ceiling_entry_point_on_tiny_subset(tmp_path):
    _experiment().main(
        [
            "ceiling",
            "--max-graphs",
            "6",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "ceiling"),
        ],
    )
    summary = pd.read_csv(tmp_path / "ceiling" / "ceiling_summary.csv")
    assert set(summary["probe"]) == {"ols", "random_forest"}
    assert summary["step"].nunique() == 9
    assert (tmp_path / "ceiling" / "ceiling_evolution.pdf").exists()


@needs_data
def test_short_time_entry_point(tmp_path):
    recon_run.main(
        [
            "short-time",
            "--quenches",
            "AFM",
            "--max-graphs",
            "10",
            "--max-nodes",
            "11",
            "--output-dir",
            str(tmp_path),
        ],
    )
    verdict = pd.read_csv(tmp_path / "short_time_verdict.csv")
    assert verdict[verdict["mandatory"]]["passed"].all()


@needs_data
def test_residual_export_on_tiny_subset():
    graphs, meta = compute_residuals("MUTAG", "AFM", max_graphs=5)
    assert len(graphs) == 5
    first = graphs[0]
    assert first.twobody_features.shape[0] == len(meta["times_ns"])
    assert first.onebody_features.shape == (len(meta["times_ns"]), first.num_nodes)


@needs_data
def test_residual_export_keeps_source_zz_scale(tmp_path):
    from qgf.data import load
    from qgf.data.io import load_dataset
    from qgf.reconstruction.residual_dataset import emit_residual_dataset

    (path,) = emit_residual_dataset("ZINC_small", "AFM", str(tmp_path), max_graphs=2)
    _, meta = load_dataset(path)
    assert meta["zz_scale_applied"] == load("zinc_afm", "test")[1]["zz_scale_applied"]


def test_retention_entry_point(tmp_path):
    rows = [
        ("random", "num_cycles", "mae", 1.0),
        ("FM_AFM", "num_cycles", "mae", 0.1),
    ]
    full = tmp_path / "full.csv"
    pd.DataFrame(rows, columns=["pe_type", "benchmark", "metric", "value"]).to_csv(
        full, index=False
    )
    residual = tmp_path / "residual.csv"
    rows[1] = ("FM_AFM", "num_cycles", "mae", 0.2)
    pd.DataFrame(rows, columns=["pe_type", "benchmark", "metric", "value"]).to_csv(
        residual,
        index=False,
    )
    _experiment().main(
        [
            "retention",
            "--residual-results",
            str(residual),
            "--full-results",
            str(full),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    table = pd.read_csv(tmp_path / "out" / "retention.csv")
    assert table["retention"].iloc[0] == pytest.approx(0.8 / 0.9)
