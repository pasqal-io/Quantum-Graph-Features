"""End-to-end sweep on injected synthetic data.

Loading the qcorr datasets is replaced by the fixture: these tests validate the
sweep mechanics (result-tensor indexing, dense/streaming equivalence,
pooled / per-graph consistency) without touching the cluster.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qgf.reconstruction.config import BasisContext, ProbeSpec, RunConfig
from qgf.reconstruction.sweep import run_per_graph, run_sweep


@pytest.fixture
def config(tmp_path) -> RunConfig:
    return RunConfig(
        dataset="TEST",
        quenches=("AFM",),
        families=("walk", "rrwp"),
        probes=(ProbeSpec(name="ols", k_grid=(1, 3)),),
        context=BasisContext(k_max=3),
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture(autouse=True)
def patched_loader(monkeypatch, tiny_samples):
    """Replace dataset loading with the synthetic graphs."""

    def loader(config, quench):
        return tiny_samples if config.max_graphs is None else tiny_samples[: config.max_graphs]

    monkeypatch.setattr("qgf.reconstruction.sweep.load_run_samples", loader)
    return loader


def test_sweep_indexes_results_by_family_k_and_time(config):
    results = run_sweep(config)
    expected_columns = {
        "dataset",
        "quench",
        "family",
        "phase",
        "k",
        "probe",
        "time_index",
        "time_ns",
        "r2",
        "r2_adjusted",
        "r2_fold_std",
        "num_columns",
        "num_parameters",
        "num_rows",
        "fold_mode",
        "intercept",
        "seed",
    }
    assert expected_columns <= set(results.columns)
    assert set(results["family"]) == {"walk", "rrwp"}
    assert set(results["k"]) == {1, 3}
    assert set(results["time_index"]) == {0, 1, 2, 3}
    assert set(results["phase"]) == {"P1"}
    assert (results["fold_mode"] == "in_sample").all()


def test_sweep_recovers_planted_signal(config):
    """The synthetic target is 0.1 A + 0.02 A^2: walk K=2 must explain everything."""
    tuned = dataclasses.replace(config, probes=(ProbeSpec(name="ols", k_grid=(1, 2)),))
    results = run_sweep(tuned)
    walk = results[(results["family"] == "walk") & (results["k"] == 2)]
    assert walk["r2"].min() > 0.999
    weaker = results[(results["family"] == "walk") & (results["k"] == 1)]
    assert weaker["r2"].max() < walk["r2"].min()


def test_sweep_restricts_to_requested_time_indices(config):
    tuned = dataclasses.replace(config, time_indices=(1, 3))
    results = run_sweep(tuned)
    assert set(results["time_index"]) == {1, 3}


def test_streaming_matches_dense_path(config):
    """The accumulated normal equations must recover the R² of the dense path."""
    dense = run_sweep(dataclasses.replace(config, streaming=False))
    streamed = run_sweep(dataclasses.replace(config, streaming=True))
    keys = ["family", "k", "time_index"]
    merged = dense.merge(streamed, on=keys, suffixes=("_dense", "_stream"))
    assert len(merged) == len(dense)
    assert np.allclose(merged["r2_dense"], merged["r2_stream"], atol=1e-8)


def test_streaming_rejects_nonlinear_probes(config):
    tuned = dataclasses.replace(
        config,
        streaming=True,
        probes=(ProbeSpec(name="random_forest", k_grid=(1,)),),
    )
    with pytest.raises(ValueError, match="Streaming"):
        run_sweep(tuned)


def test_streaming_rejects_time_aligned_families(config):
    tuned = dataclasses.replace(config, streaming=True, families=("twobody",))
    with pytest.raises(NotImplementedError, match="streaming"):
        run_sweep(tuned)


def test_intercept_adds_one_effective_parameter(config):
    without = run_sweep(config)
    with_constant = run_sweep(dataclasses.replace(config, intercept=True))
    reference = without[(without["family"] == "walk") & (without["k"] == 1)]
    augmented = with_constant[(with_constant["family"] == "walk") & (with_constant["k"] == 1)]
    assert augmented["num_columns"].iloc[0] == reference["num_columns"].iloc[0] + 1
    assert augmented["r2"].iloc[0] >= reference["r2"].iloc[0] - 1e-12


def test_time_aligned_family_uses_one_column_per_step(config):
    tuned = dataclasses.replace(
        config,
        families=("twobody",),
        probes=(ProbeSpec(name="ols", k_grid=(1,)),),
        context=BasisContext(k_max=1, times_ns=(100.0, 200.0, 300.0, 400.0)),
    )
    results = run_sweep(tuned)
    assert (results["num_columns"] == 1).all()
    assert set(results["time_index"]) == {0, 1, 2, 3}


def test_nonlinear_probe_uses_graph_folds(config):
    tuned = dataclasses.replace(
        config,
        families=("walk",),
        probes=(ProbeSpec(name="random_forest", k_grid=(2,), time_indices=(0,)),),
        n_splits=2,
    )
    results = run_sweep(tuned)
    assert (results["fold_mode"] == "graph").all()
    assert set(results["time_index"]) == {0}


def test_per_graph_run_uses_cayley_hamilton_order(config):
    per_graph = run_per_graph(config, family="walk")
    assert set(per_graph.columns) >= {"graph_id", "num_nodes", "k", "r2", "time_ns"}
    assert (per_graph["k"] == per_graph["num_nodes"] - 1).all()
    # The planted target is in the span from K=2, hence R² ~ 1 per graph.
    assert per_graph["r2"].min() > 0.999


def test_sweep_rejects_k_beyond_k_max(config):
    """A K above k_max would be silently truncated: the sweep refuses."""
    tuned = dataclasses.replace(
        config,
        probes=(ProbeSpec(name="ols", k_grid=(1, 99)),),
    )
    with pytest.raises(ValueError, match="k_max"):
        run_sweep(tuned)


def test_cache_preserves_time_alignment(config):
    """A design read back from the disk cache keeps its time alignment (original bug)."""
    tuned = dataclasses.replace(
        config,
        families=("twobody",),
        probes=(ProbeSpec(name="ols", k_grid=(1,)),),
        context=BasisContext(k_max=1, times_ns=(100.0, 200.0, 300.0, 400.0)),
    )
    first = run_sweep(tuned)
    second = run_sweep(tuned)
    assert (second["num_columns"] == 1).all()
    assert np.allclose(first["r2"], second["r2"])


def test_random_forest_reports_cross_fold_spread(config):
    tuned = dataclasses.replace(
        config,
        families=("walk",),
        probes=(ProbeSpec(name="random_forest", k_grid=(2,), time_indices=(0,)),),
        n_splits=2,
    )
    results = run_sweep(tuned)
    assert np.isfinite(results["r2_fold_std"]).all()
