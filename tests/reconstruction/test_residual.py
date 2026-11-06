"""Residual export, residual characterisation helpers and residual analyses."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from qgf.reconstruction import analysis
from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.design import (
    build_pooled_design,
    normalize_columns,
    pooled_groups,
    pooled_targets,
    with_constant,
)
from qgf.reconstruction.residual import ceiling_residual, group_demean
from qgf.reconstruction.residual_dataset import (
    node_ceiling_design,
    node_residuals,
    pooled_residual,
    scatter_to_tensor,
)

from .conftest import make_sample


def _design(samples, family="walk+rrwp"):
    block = build_pooled_design(samples, family, BasisContext(k_max=3), "TEST", cache_dir=None)
    return normalize_columns(with_constant(block).matrix)


def test_pooled_residual_is_orthogonal_to_the_design(tiny_samples):
    matrix = _design(tiny_samples)
    rng = np.random.default_rng(0)
    targets = pooled_targets(tiny_samples) + rng.normal(scale=0.01, size=(matrix.shape[0], 4))
    residual = pooled_residual(matrix, targets)
    assert np.allclose(matrix.T @ residual, 0.0, atol=1e-10)
    # Normal equations and direct least squares give the same residual.
    assert np.allclose(residual[:, 0], ceiling_residual(matrix, targets[:, 0]), atol=1e-10)


def test_planted_signal_leaves_zero_residual(tiny_samples):
    """The synthetic target 0.1 A + 0.02 A^2 is in the walk span."""
    residual = pooled_residual(_design(tiny_samples), pooled_targets(tiny_samples))
    assert np.abs(residual).max() < 1e-10


def test_scatter_to_tensor_is_symmetric_with_zero_diagonal(tiny_samples):
    sample = tiny_samples[0]
    values = np.arange(sample.num_pairs * 2, dtype=float).reshape(sample.num_pairs, 2)
    tensor = scatter_to_tensor(values, sample.num_nodes, sample.pair_index)
    assert tensor.shape == (2, sample.num_nodes, sample.num_nodes)
    assert np.allclose(tensor, np.swapaxes(tensor, 1, 2))
    assert np.all(np.diagonal(tensor, axis1=1, axis2=2) == 0)
    rows, cols = sample.pair_index
    assert np.allclose(tensor[1][rows, cols], values[:, 1])


def test_node_residuals_remove_degree_signal(tiny_samples):
    context = BasisContext(k_max=3)
    for sample in tiny_samples:
        degree = np.asarray(sample.adjacency).sum(axis=1)
        sample.occupations = np.stack([0.1 * degree, 0.2 * degree + 0.05], axis=1)
    assert node_ceiling_design(tiny_samples[0], context).shape[0] == tiny_samples[0].num_nodes
    residuals = node_residuals(tiny_samples, context)
    assert len(residuals) == len(tiny_samples)
    assert max(np.abs(r).max() for r in residuals) < 1e-7


def test_node_residuals_need_one_body_features(tiny_samples):
    assert node_residuals(tiny_samples, BasisContext(k_max=3)) is None


def test_group_demean_zeroes_per_graph_means(tiny_samples):
    groups = pooled_groups(tiny_samples)
    values = np.random.default_rng(1).normal(size=groups.size)
    demeaned = group_demean(values, groups)
    for graph in np.unique(groups):
        assert demeaned[groups == graph].mean() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------


@pytest.fixture
def ring_samples():
    graphs = [nx.cycle_graph(6), nx.path_graph(7), nx.cycle_graph(8), nx.ladder_graph(4)]
    samples = [make_sample(graph, index) for index, graph in enumerate(graphs)]
    rng = np.random.default_rng(2)
    residuals = []
    for sample in samples:
        noise = rng.normal(scale=0.01, size=sample.correlations.shape)
        noise = noise + np.swapaxes(noise, 1, 2)
        for step in range(noise.shape[0]):
            np.fill_diagonal(noise[step], 0.0)
        residuals.append(noise)
    return samples, residuals


def test_profiles_have_monotone_energy_cdf(ring_samples):
    samples, residuals = ring_samples
    profiles = analysis.spectral_range_profiles(samples, residuals, time_index=1)
    for source in analysis.SOURCES:
        cdf = profiles["energy_cdf"][source]
        assert np.all(np.diff(cdf) >= -1e-12)
        assert cdf[-1] == pytest.approx(1.0)
    # The planted target is short-range: nearest neighbours dominate the initial profile.
    initial = profiles["dist_mean"]["initial"]
    assert initial[1] > initial[3]
    tables = analysis.profiles_table({"AFM": profiles}, time_index=1)
    assert set(tables) == {"energy_cdf", "range", "centroid"}


def test_rank_test_detects_a_uniform_component(ring_samples):
    samples, _ = ring_samples
    uniform = []
    for sample in samples:
        tensor = np.ones((4, sample.num_nodes, sample.num_nodes))
        for step in range(4):
            np.fill_diagonal(tensor[step], 0.0)
        uniform.append(-0.01 * tensor)
    result = analysis.rank_test(samples, uniform, time_index=0)
    assert result["uniform_overlap_median"] > 0.99
    assert result["tail_kept_after_demean"] == pytest.approx(0.0, abs=1e-9)


def test_retention_is_normalised_between_random_and_full():
    rows = []
    for pe, mae, acc in (("random", 1.0, 0.5), ("FM_AFM", 0.2, 0.9)):
        rows.append(
            {"pe_type": pe, "benchmark": "degree_prediction", "metric": "mae", "value": mae}
        )
        rows.append(
            {"pe_type": pe, "benchmark": "link_prediction", "metric": "accuracy", "value": acc},
        )
    full = pd.DataFrame(rows)
    residual = full.copy()
    residual.loc[residual["pe_type"] == "FM_AFM", "value"] = [0.6, 0.7]
    table = analysis.retention(residual, full)
    by_bench = table.set_index("benchmark")["retention"]
    assert by_bench["degree_prediction"] == pytest.approx(0.5)
    assert by_bench["link_prediction"] == pytest.approx(0.5)
    summary = analysis.retention_summary(table)
    assert set(summary["category"]) == {"Node", "Edge"}
