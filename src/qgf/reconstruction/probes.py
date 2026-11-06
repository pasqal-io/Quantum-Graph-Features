"""Reconstruction probes: pooled OLS, Nystroem kernel ridge, random forest.

The linear probe is an OLS solved by Moore-Penrose pseudo-inverse and evaluated
**in-sample**: we look for an expressivity ceiling, not generalisation, and with P = 30 505
rows for at most a few hundred parameters overfitting is negligible. Nonlinear probes are
scored held-out over 5 folds, at the **graph** level (all pairs of a graph in the same fold, no
information leak between folds).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

LOGGER = logging.getLogger(__name__)

#: Hyperparameters of the Nystroem kernel ridge.
NYSTROEM_COMPONENTS: int = 500
KERNEL_RIDGE_ALPHA: float = 1e-3
#: Subsample size of the median-distance bandwidth heuristic.
BANDWIDTH_SAMPLE: int = 1000

#: Hyperparameters of the random forest.
RF_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 20
RF_MIN_SAMPLES_LEAF: int = 10


@dataclass(frozen=True)
class ProbeResult:
    """R^2 per time step and effective model size.

    ``fold_r2`` is (n_folds, n_steps) for held-out probes (None for in-sample OLS): it
    gives the cross-fold spread used as the reproduction tolerance.
    """

    r2: np.ndarray
    num_columns: int
    num_parameters: int
    num_rows: int
    fold_r2: np.ndarray | None = None

    @property
    def r2_adjusted(self) -> np.ndarray:
        """R^2 adjusted for the number of effective parameters."""
        denominator = self.num_rows - self.num_parameters - 1
        if denominator <= 0:
            return np.full_like(self.r2, np.nan)
        return 1.0 - (1.0 - self.r2) * (self.num_rows - 1) / denominator

    @property
    def fold_std(self) -> np.ndarray:
        """Cross-fold standard deviation of R^2 per time step (NaN for OLS)."""
        if self.fold_r2 is None:
            return np.full_like(self.r2, np.nan)
        return self.fold_r2.std(axis=0)


def _r2_from_sums(ss_res: np.ndarray, ss_tot: np.ndarray) -> np.ndarray:
    """R^2 = 1 - SS_res / SS_tot, NaN when the target is constant."""
    ss_res = np.where(ss_res < 0.0, 0.0, ss_res)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return np.where(ss_tot > 0.0, r2, np.nan)


def _r2(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    residual = targets - predictions
    ss_res = np.einsum("pt,pt->t", residual, residual)
    centered = targets - targets.mean(axis=0, keepdims=True)
    ss_tot = np.einsum("pt,pt->t", centered, centered)
    return _r2_from_sums(ss_res, ss_tot)


def _fold_r2(
    targets: np.ndarray,
    predictions: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    return np.stack([_r2(targets[test], predictions[test]) for _, test in folds], axis=0)


def pooled_r2(matrix: np.ndarray, targets: np.ndarray) -> ProbeResult:
    """In-sample pooled OLS, all time steps in one solve.

    Args:
        matrix: (P, n_cols) pooled design, already column-normalised.
        targets: (P,) or (P, N_t) pooled target.

    R^2 is computed against the pooled mean ybar(t) = P^-1 sum_p y_p(t).
    """
    y = np.atleast_2d(targets.T).T if targets.ndim == 1 else targets
    beta, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ beta
    ss_res = np.einsum("pt,pt->t", residual, residual)
    centered = y - y.mean(axis=0, keepdims=True)
    ss_tot = np.einsum("pt,pt->t", centered, centered)
    return ProbeResult(
        r2=_r2_from_sums(ss_res, ss_tot),
        num_columns=int(matrix.shape[1]),
        num_parameters=int(rank),
        num_rows=int(matrix.shape[0]),
    )


class GramAccumulator:
    """Block accumulation of the normal equations X^T X beta = X^T y.

    For ZINC (P ~ 3e6) the pooled design costs 1.2 GB per K value; the Gram matrix stays
    (K+1)^2.
    """

    def __init__(self, num_columns: int, num_targets: int) -> None:
        self.gram = np.zeros((num_columns, num_columns))
        self.moment = np.zeros((num_columns, num_targets))
        self.target_sum = np.zeros(num_targets)
        self.target_square_sum = np.zeros(num_targets)
        self.num_rows = 0

    def add(self, matrix: np.ndarray, targets: np.ndarray) -> None:
        """Adds a block of rows (one graph or a batch of graphs)."""
        self.gram += matrix.T @ matrix
        self.moment += matrix.T @ targets
        self.target_sum += targets.sum(axis=0)
        self.target_square_sum += np.einsum("pt,pt->t", targets, targets)
        self.num_rows += int(matrix.shape[0])

    def result(self) -> ProbeResult:
        """Solves and returns the pooled R^2 per time step."""
        beta, _, rank, _ = np.linalg.lstsq(self.gram, self.moment, rcond=None)
        ss_res = (
            self.target_square_sum
            - 2.0 * np.einsum("ct,ct->t", beta, self.moment)
            + np.einsum("ct,ct->t", beta, self.gram @ beta)
        )
        ss_tot = self.target_square_sum - self.target_sum**2 / max(self.num_rows, 1)
        return ProbeResult(
            r2=_r2_from_sums(ss_res, ss_tot),
            num_columns=int(self.gram.shape[0]),
            num_parameters=int(rank),
            num_rows=self.num_rows,
        )


def per_graph_r2(
    matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> np.ndarray:
    """Graph-by-graph OLS R^2: (n_graphs, N_t) (Cayley-Hamilton test, K = n - 1)."""
    return np.stack(
        [pooled_r2(matrix, target).r2 for matrix, target in zip(matrices, targets)],
        axis=0,
    )


# --------------------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------------------


def fold_indices(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Graph-level fold split: all pairs of a graph fall in the same fold."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = rng.permutation(unique)
    folds = []
    for chunk in np.array_split(shuffled, n_splits):
        test_mask = np.isin(groups, chunk)
        folds.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return folds


def _median_bandwidth(matrix: np.ndarray, seed: int) -> float:
    """Median pairwise-distance heuristic, on a subsample."""
    rng = np.random.default_rng(seed)
    size = min(BANDWIDTH_SAMPLE, matrix.shape[0])
    rows = rng.choice(matrix.shape[0], size=size, replace=False)
    sample = matrix[rows]
    distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    upper = distances[np.triu_indices(size, k=1)]
    median = float(np.median(upper))
    return median if median > 0 else 1.0


def kernel_ridge_cv(
    matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    n_components: int = NYSTROEM_COMPONENTS,
    alpha: float = KERNEL_RIDGE_ALPHA,
) -> ProbeResult:
    """RBF kernel ridge approximated by Nystroem, scored held-out.

    The transform does not depend on the target: it is fitted once per fold and shared by
    all time steps.
    """
    bandwidth = _median_bandwidth(matrix, seed)
    gamma = 1.0 / (2.0 * bandwidth**2)
    predictions = np.empty_like(targets)
    folds = fold_indices(groups, n_splits, seed)
    for fold, (train, test) in enumerate(folds):
        transform = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=min(n_components, train.size),
            random_state=seed + fold,
        )
        features_train = transform.fit_transform(matrix[train])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(features_train, targets[train])
        predicted = np.asarray(model.predict(transform.transform(matrix[test])))
        predictions[test] = predicted.reshape(test.size, targets.shape[1])
    return ProbeResult(
        r2=_r2(targets, predictions),
        num_columns=int(matrix.shape[1]),
        num_parameters=int(min(n_components, matrix.shape[0])),
        num_rows=int(matrix.shape[0]),
        fold_r2=_fold_r2(targets, predictions, folds),
    )


def random_forest_cv(
    matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    n_jobs: int = -1,
) -> ProbeResult:
    """Random forest scored held-out, one independent forest per time step."""
    predictions = np.empty_like(targets)
    folds = fold_indices(groups, n_splits, seed)
    for step in range(targets.shape[1]):
        for fold, (train, test) in enumerate(folds):
            model = RandomForestRegressor(
                n_estimators=RF_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=seed + fold,
                n_jobs=n_jobs,
            )
            model.fit(matrix[train], targets[train, step])
            predictions[test, step] = model.predict(matrix[test])
    return ProbeResult(
        r2=_r2(targets, predictions),
        num_columns=int(matrix.shape[1]),
        num_parameters=int(matrix.shape[1]),
        num_rows=int(matrix.shape[0]),
        fold_r2=_fold_r2(targets, predictions, folds),
    )


def permutation_group_importance(
    matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    column_groups: dict[str, Sequence[int]],
    n_splits: int = 5,
    seed: int = 0,
    n_repeats: int = 3,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    """Grouped permutation importance of columns (held-out R^2 drop per column group)."""
    folds = fold_indices(groups, n_splits, seed)
    num_steps = targets.shape[1]
    baseline = np.empty_like(targets)
    permuted = {name: np.zeros((targets.shape[0], num_steps)) for name in column_groups}
    rng = np.random.default_rng(seed)

    for step in range(num_steps):
        for fold, (train, test) in enumerate(folds):
            model = RandomForestRegressor(
                n_estimators=RF_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=seed + fold,
                n_jobs=n_jobs,
            )
            model.fit(matrix[train], targets[train, step])
            baseline[test, step] = model.predict(matrix[test])
            for name, columns in column_groups.items():
                repeats = np.zeros(test.size)
                for _ in range(n_repeats):
                    shuffled = matrix[test].copy()
                    order = rng.permutation(test.size)
                    shuffled[:, list(columns)] = shuffled[order][:, list(columns)]
                    repeats += model.predict(shuffled)
                permuted[name][test, step] = repeats / n_repeats

    base_r2 = _r2(targets, baseline)
    importances = {
        name: base_r2 - _r2(targets, prediction) for name, prediction in permuted.items()
    }
    importances["_baseline_r2"] = base_r2
    return importances


PROBE_FACTORIES: dict[str, Callable[..., ProbeResult]] = {
    "kernel_ridge": kernel_ridge_cv,
    "random_forest": random_forest_cv,
}
