"""Extension bases: register geometry, interaction matrix, physical controls.

The guiding idea is to test descriptors **closer and closer to the actual Hamiltonian**,
to locate the boundary between what is structural and what is genuinely quantum.
"""

from __future__ import annotations

import numpy as np

from qgf.reconstruction.classical import twa_correlations
from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import ORDER_ALWAYS, DesignBlock, register_basis
from qgf.reconstruction.features_base import laplacian_features, walk_powers
from qgf.reconstruction.quench import (
    interaction_matrix,
    nearest_neighbour_coupling,
    two_body_correlations,
)

#: Direct and inverse powers of the distance in the geometry basis.
DIRECT_POWERS: tuple[int, ...] = (1, 2)
INVERSE_POWERS: tuple[int, ...] = (1, 2, 3, 6, 12)


def _require_positions(sample: GraphSample) -> np.ndarray:
    if sample.embedded_position is None:
        raise ValueError(f"Graph {sample.graph_id}: no atom positions, geometry unavailable")
    return np.asarray(sample.embedded_position, dtype=float)


def _pair_distances(positions: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rows, cols = pairs
    return np.linalg.norm(positions[rows] - positions[cols], axis=-1)


def radial_basis(distances: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """Gaussian radial expansion exp[-(d - mu_c)^2 / 2 sigma^2]."""
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2.0 * width**2))


@register_basis("geometry", phase="P2")
def geometry_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Geometric distances of the atom layout, in units of r_NN (diagnostic block)."""
    positions = _require_positions(sample)
    rows, cols = sample.pair_index
    r_nn = context.r_nn_um
    reduced = _pair_distances(positions, sample.pair_index) / r_nn

    columns: list[np.ndarray] = []
    labels: list[str] = []
    for power in DIRECT_POWERS:
        columns.append(reduced**power)
        labels.append(f"geo_d{power}")

    positive = reduced > 0
    for power in INVERSE_POWERS:
        inverse = np.zeros_like(reduced)
        inverse[positive] = reduced[positive] ** (-power)
        columns.append(inverse)
        labels.append(f"geo_inv{power}")

    low, high = context.rbf_range_rnn
    centers = np.linspace(low, high, context.rbf_centers)
    width = float(centers[1] - centers[0]) if context.rbf_centers > 1 else 1.0
    expansion = radial_basis(reduced, centers, width)
    for index in range(expansion.shape[1]):
        columns.append(expansion[:, index])
        labels.append(f"geo_rbf_{index:02d}")

    # Geometric neighbourhood statistics, in units of r_NN.
    full = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) / r_nn
    off_diagonal = full + np.diag(np.full(full.shape[0], np.inf))
    min_distance = off_diagonal.min(axis=1)
    with np.errstate(divide="ignore"):
        density = np.where(off_diagonal > 0, off_diagonal, np.inf) ** -6
    local_density = density.sum(axis=1)
    for name, values in (("mindist", min_distance), ("density", local_density)):
        columns.append(values[rows])
        labels.append(f"geo_{name}_i")
        columns.append(values[cols])
        labels.append(f"geo_{name}_j")

    matrix = np.stack(columns, axis=1)
    return DesignBlock(
        matrix=matrix,
        labels=tuple(labels),
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# Interaction matrix
# --------------------------------------------------------------------------------------


def scaled_interaction(sample: GraphSample, context: BasisContext) -> np.ndarray:
    """Dimensionless J/J_NN, exactly what enters the Hamiltonian.

    Unlike A, J is weighted and never exactly zero off bonds: that tail is precisely what
    the walk basis cannot represent.
    """
    positions = _require_positions(sample)
    couplings = interaction_matrix(positions, c6=context.c6)
    return couplings / nearest_neighbour_coupling(r_nn=context.r_nn_um, c6=context.c6)


@register_basis("interaction", phase="P2")
def interaction_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Interaction matrix, one column: J~_ij = (r_NN / d_ij)^6."""
    couplings = scaled_interaction(sample, context)
    rows, cols = sample.pair_index
    return DesignBlock(
        matrix=couplings[rows, cols][:, None],
        labels=("interaction",),
        orders=np.array([ORDER_ALWAYS], dtype=np.int64),
    )


@register_basis("distance", phase="P2")
def distance_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """2D register distance, one column: d_ij / r_NN."""
    positions = _require_positions(sample)
    reduced = _pair_distances(positions, sample.pair_index) / context.r_nn_um
    return DesignBlock(
        matrix=reduced[:, None],
        labels=("distance",),
        orders=np.array([ORDER_ALWAYS], dtype=np.int64),
    )


@register_basis("jpow", phase="P2")
def interaction_power_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Powers (J~^k)_ij for k = 1..k_max, weighted analogue of the walk basis."""
    couplings = scaled_interaction(sample, context)
    powers = walk_powers(couplings, context.k_max, sample.pair_index)
    return DesignBlock(
        matrix=np.ascontiguousarray(powers[:, 1:]),
        labels=tuple(f"jpow_{k}" for k in range(1, context.k_max + 1)),
        orders=np.arange(1, context.k_max + 1, dtype=np.int64),
    )


@register_basis("jrrwp", phase="P2")
def interaction_rrwp_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Powers ((D_J^-1 J~)^k)_ij, weighted analogue of RRWP."""
    couplings = scaled_interaction(sample, context)
    degree = couplings.sum(axis=1, keepdims=True)
    stochastic = couplings / np.maximum(degree, 1e-30)
    powers = walk_powers(stochastic, context.k_max, sample.pair_index)
    return DesignBlock(
        matrix=np.ascontiguousarray(powers[:, 1:]),
        labels=tuple(f"jrrwp_{k}" for k in range(1, context.k_max + 1)),
        orders=np.arange(1, context.k_max + 1, dtype=np.int64),
    )


@register_basis("jlaplacian", phase="P2")
def interaction_laplacian_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Spectrum of the weighted Laplacian L_J = I - D_J^-1/2 J~ D_J^-1/2."""
    couplings = scaled_interaction(sample, context)
    k_max = context.k_max
    matrix = laplacian_features(couplings, k_max, sample.pair_index, context.sign_tie_decimals)
    labels = (
        tuple(f"jlap_ui_{k}" for k in range(1, k_max + 1))
        + tuple(f"jlap_uj_{k}" for k in range(1, k_max + 1))
        + tuple(f"jlap_lambda_{k}" for k in range(1, k_max + 1))
    )
    return DesignBlock(
        matrix=matrix,
        labels=labels,
        orders=np.tile(np.arange(1, k_max + 1, dtype=np.int64), 3),
    )


# --------------------------------------------------------------------------------------
# Physical controls
# --------------------------------------------------------------------------------------


def susceptibility_betas(couplings: np.ndarray, context: BasisContext) -> np.ndarray:
    """Log grid of beta in the paramagnetic regime beta < 1/lambda_max."""
    spectral_radius = float(np.abs(np.linalg.eigvalsh(couplings)).max())
    fractions = np.logspace(-2.0, np.log10(context.beta_max_fraction), context.beta_count)
    if context.beta_mode == "relative":
        return fractions / max(spectral_radius, 1e-30)
    # Absolute mode: common grid, clipped below the critical beta of the graph.
    return np.minimum(fractions, context.beta_max_fraction / max(spectral_radius, 1e-30))


@register_basis("meanfield", phase="P2")
def mean_field_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Mean-field linear response chi = (1 - beta J~)^-1.

    Exact resummation of all powers sum_k beta^k J~^k: an infinite-order weighted walk
    basis compressed into one column per beta.
    """
    couplings = scaled_interaction(sample, context)
    identity = np.eye(couplings.shape[0])
    rows, cols = sample.pair_index
    betas = susceptibility_betas(couplings, context)
    columns = [np.linalg.solve(identity - beta * couplings, identity)[rows, cols] for beta in betas]
    matrix = np.stack(columns, axis=1)
    return DesignBlock(
        matrix=matrix,
        labels=tuple(f"chi_beta{index:02d}" for index in range(betas.size)),
        orders=np.full(matrix.shape[1], ORDER_ALWAYS, dtype=np.int64),
    )


def _two_body_targets(sample: GraphSample, context: BasisContext) -> np.ndarray:
    """(n_pairs, N_t) exact correlation of the two-atom system with coupling J_ij."""
    positions = _require_positions(sample)
    couplings = interaction_matrix(positions, c6=context.c6)
    rows, cols = sample.pair_index
    j_nn = nearest_neighbour_coupling(r_nn=context.r_nn_um, c6=context.c6)
    omega = context.quench.omega(j_nn)
    delta = context.quench.delta(couplings)
    times_us = np.asarray(context.times_ns, dtype=float) / 1000.0
    return two_body_correlations(couplings[rows, cols], times_us, omega, delta)


@register_basis("twobody", phase="P2", quench_sensitive=True)
def two_body_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Exact two-atom correlation, aligned with the target time.

    One column is active per time step: the one of the target instant.
    """
    values = _two_body_targets(sample, context)
    num_steps = values.shape[1]
    return DesignBlock(
        matrix=values,
        labels=tuple(f"twobody_t{step}" for step in range(num_steps)),
        orders=np.full(num_steps, ORDER_ALWAYS, dtype=np.int64),
        time_index=np.arange(num_steps, dtype=np.int64),
    )


@register_basis("twobody_curve", phase="P2", quench_sensitive=True)
def two_body_curve_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Whole two-atom curve as static columns (more permissive variant)."""
    values = _two_body_targets(sample, context)
    num_steps = values.shape[1]
    return DesignBlock(
        matrix=values,
        labels=tuple(f"twobody_curve_t{step}" for step in range(num_steps)),
        orders=np.full(num_steps, ORDER_ALWAYS, dtype=np.int64),
    )


@register_basis("classical", phase="P2", quench_sensitive=True)
def classical_dynamics_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Classical spin-dynamics (dTWA) correlation, aligned with the target time.

    Classical limit of the same quench: whatever survives it is what quantum dynamics
    adds beyond the classical many-body limit.
    """
    positions = _require_positions(sample)
    couplings = interaction_matrix(positions, c6=context.c6)
    rows, cols = sample.pair_index
    j_nn = nearest_neighbour_coupling(r_nn=context.r_nn_um, c6=context.c6)
    omega = context.quench.omega(j_nn)
    delta = context.quench.delta(couplings)
    times_us = np.asarray(context.times_ns, dtype=float) / 1000.0
    correlations = twa_correlations(
        couplings,
        times_us,
        omega,
        delta,
        num_trajectories=context.twa_trajectories,
        dt_us=context.twa_dt_us,
        seed=context.twa_seed,
    )
    values = correlations[:, rows, cols].T  # (n_pairs, N_t)
    num_steps = values.shape[1]
    return DesignBlock(
        matrix=values,
        labels=tuple(f"classical_t{step}" for step in range(num_steps)),
        orders=np.full(num_steps, ORDER_ALWAYS, dtype=np.int64),
        time_index=np.arange(num_steps, dtype=np.int64),
    )
