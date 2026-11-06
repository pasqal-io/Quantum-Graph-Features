"""Probes: pooled OLS, streaming equivalence, folds, non-linear probes."""

from __future__ import annotations

import numpy as np
import pytest

from qgf.reconstruction.probes import (
    GramAccumulator,
    ProbeResult,
    fold_indices,
    kernel_ridge_cv,
    per_graph_r2,
    permutation_group_importance,
    pooled_r2,
    random_forest_cv,
)


def _linear_problem(num_rows: int = 400, num_columns: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(num_rows, num_columns))
    beta = rng.normal(size=(num_columns, 3))
    return matrix, matrix @ beta


def test_pooled_r2_is_one_on_exact_fit():
    matrix, targets = _linear_problem()
    result = pooled_r2(matrix, targets)
    assert np.allclose(result.r2, 1.0)
    assert result.num_rows == matrix.shape[0]
    assert result.num_parameters == matrix.shape[1]


def test_pooled_r2_matches_closed_form_on_single_column():
    """R² without intercept, computed against the pooled mean."""
    matrix = np.array([[1.0], [2.0], [3.0], [4.0]])
    targets = np.array([[1.2], [1.9], [3.2], [3.9]])
    beta = float(np.linalg.lstsq(matrix, targets, rcond=None)[0].item())
    residual = targets - matrix * beta
    expected = 1.0 - float((residual.T @ residual).item()) / float(
        ((targets - targets.mean()) ** 2).sum(),
    )
    assert pooled_r2(matrix, targets).r2[0] == pytest.approx(expected)


def test_pooled_r2_accepts_one_dimensional_target():
    matrix, targets = _linear_problem()
    assert pooled_r2(matrix, targets[:, 0]).r2.shape == (1,)


def test_pooled_r2_is_nan_on_constant_target():
    matrix = np.array([[1.0], [2.0]])
    targets = np.array([[3.0], [3.0]])
    assert np.isnan(pooled_r2(matrix, targets).r2[0])


def test_pooled_r2_handles_identically_zero_column():
    """The k=0 column of the walk basis is zero: the lstsq must stay well-behaved."""
    matrix, targets = _linear_problem()
    padded = np.concatenate([np.zeros((matrix.shape[0], 1)), matrix], axis=1)
    result = pooled_r2(padded, targets)
    assert np.allclose(result.r2, 1.0)
    assert result.num_parameters == matrix.shape[1]  # rang, pas nombre de colonnes


def test_gram_accumulator_matches_dense_ols():
    """The streaming path (normal equations) must equal the dense path."""
    matrix, targets = _linear_problem(num_rows=600, num_columns=5, seed=3)
    targets = targets + 0.1 * np.random.default_rng(4).normal(size=targets.shape)
    dense = pooled_r2(matrix, targets)
    accumulator = GramAccumulator(matrix.shape[1], targets.shape[1])
    for start in range(0, matrix.shape[0], 97):
        accumulator.add(matrix[start : start + 97], targets[start : start + 97])
    streamed = accumulator.result()
    assert streamed.num_rows == dense.num_rows
    assert np.allclose(streamed.r2, dense.r2, atol=1e-8)


def test_r2_adjusted_penalises_columns():
    result = ProbeResult(r2=np.array([0.9]), num_columns=10, num_parameters=10, num_rows=100)
    expected = 1.0 - 0.1 * 99 / 89
    assert result.r2_adjusted[0] == pytest.approx(expected)
    assert result.r2_adjusted[0] < result.r2[0]


def test_per_graph_r2_returns_one_row_per_graph():
    matrices = [_linear_problem(num_rows=50, seed=index)[0] for index in range(3)]
    targets = [matrix @ np.ones((matrix.shape[1], 2)) for matrix in matrices]
    scores = per_graph_r2(matrices, targets)
    assert scores.shape == (3, 2)
    assert np.allclose(scores, 1.0)


def test_graph_folds_never_split_a_graph():
    """The per-graph split is the reference: no pair from the same graph on both
    sides."""
    groups = np.repeat(np.arange(20), 7)
    folds = fold_indices(groups, n_splits=5, seed=0)
    assert len(folds) == 5
    covered = []
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        covered.extend(test.tolist())
    assert sorted(covered) == list(range(groups.size))


def _learnable_problem(num_graphs: int = 30, per_graph: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(num_graphs), per_graph)
    matrix = rng.normal(size=(groups.size, 3))
    signal = np.sin(matrix[:, 0]) + matrix[:, 1] ** 2
    targets = np.stack([signal, 2.0 * signal], axis=1)
    return matrix, targets, groups


def test_kernel_ridge_recovers_smooth_signal():
    matrix, targets, groups = _learnable_problem()
    result = kernel_ridge_cv(matrix, targets, groups, n_splits=3, n_components=50, seed=0)
    assert result.r2.shape == (2,)
    assert result.r2.min() > 0.5


def test_random_forest_recovers_smooth_signal():
    matrix, targets, groups = _learnable_problem()
    result = random_forest_cv(matrix, targets, groups, n_splits=3, seed=0, n_jobs=1)
    assert result.r2.min() > 0.5


def test_permutation_importance_ranks_informative_group():
    """The grouped permutation must degrade the informative group more."""
    matrix, targets, groups = _learnable_problem(num_graphs=20, per_graph=15)
    importances = permutation_group_importance(
        matrix,
        targets[:, :1],
        groups,
        {"informatif": [0, 1], "bruit": [2]},
        n_splits=3,
        seed=0,
        n_repeats=2,
        n_jobs=1,
    )
    assert importances["informatif"][0] > importances["bruit"][0]
    assert importances["_baseline_r2"][0] > 0.5


@pytest.mark.parametrize("num_targets", [1, 3])
def test_nonlinear_probes_accept_a_single_target_column(num_targets):
    """Ridge flattens its output on a single-column target: the probe must withstand it."""
    matrix, targets, groups = _learnable_problem(num_graphs=15, per_graph=10)
    restricted = np.repeat(targets[:, :1], num_targets, axis=1)
    kernel = kernel_ridge_cv(matrix, restricted, groups, n_splits=3, n_components=40, seed=0)
    forest = random_forest_cv(matrix, restricted, groups, n_splits=3, seed=0, n_jobs=1)
    assert kernel.r2.shape == (num_targets,)
    assert forest.r2.shape == (num_targets,)
    assert np.isfinite(kernel.r2).all()
