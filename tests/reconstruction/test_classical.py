"""Dynamique de spins classiques (dTWA) : physique et invariants."""

from __future__ import annotations

import numpy as np
import pytest

from qgf.reconstruction.classical import (
    _connected,
    effective_field,
    initial_spins,
    twa_correlations,
)

TIMES_US = np.array([0.0, 0.05, 0.1, 0.2])


def test_initial_spins_reproduce_the_vacuum():
    """|0...0> : s^z = +1 (zero occupation), s^{x,y} centred with unit variance."""
    rng = np.random.default_rng(0)
    spins = initial_spins(4, 5000, rng)
    assert spins.shape == (5000, 4, 3)
    assert np.all(spins[..., 2] == 1.0)
    assert np.allclose(spins[..., 0].mean(), 0.0, atol=0.03)
    assert np.allclose(spins[..., 0].var(), 1.0, atol=0.05)
    # Initial occupation is zero on every trajectory.
    assert np.all(0.5 * (1.0 - spins[..., 2]) == 0.0)


def test_effective_field_matches_hamiltonian_gradient():
    """B^x = Ω/2, B^z = δ/2 - (1/2) Σ_j J_ij n_j, B^y = 0."""
    couplings = np.array([[0.0, 2.0], [2.0, 0.0]])
    spins = np.array([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])  # n = (1, 0)
    field = effective_field(spins, couplings, omega=3.0, delta=1.0)
    # Atome 0 : B^z = 0.5 - 0.5 * J_01 * n_1 = 0.5 - 0 = 0.5
    assert field[0, 0] == pytest.approx([1.5, 0.0, 0.5])
    # Atome 1 : B^z = 0.5 - 0.5 * J_10 * n_0 = 0.5 - 0.5*2*1 = -0.5
    assert field[0, 1] == pytest.approx([1.5, 0.0, -0.5])


def test_no_drive_freezes_occupations():
    """Ω = 0: the field is along z, s^z is conserved, no connected correlation."""
    couplings = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
    correlations = twa_correlations(
        couplings, TIMES_US, omega=0.0, delta=2.0, num_trajectories=64, dt_us=0.02
    )
    assert np.allclose(correlations, 0.0, atol=1e-12)


def test_initial_time_has_zero_connected_correlation():
    """At t = 0 the occupation is zero and deterministic: C_ij(0) = 0."""
    couplings = np.array([[0.0, 3.0], [3.0, 0.0]])
    correlations = twa_correlations(
        couplings, TIMES_US, omega=10.0, delta=0.0, num_trajectories=128, dt_us=0.01
    )
    assert np.allclose(correlations[0], 0.0, atol=1e-12)


def test_correlations_are_symmetric_and_zero_diagonal():
    couplings = np.array([[0.0, 4.0, 1.0], [4.0, 0.0, 4.0], [1.0, 4.0, 0.0]])
    correlations = twa_correlations(
        couplings, TIMES_US, omega=8.0, delta=1.0, num_trajectories=256, dt_us=0.01, seed=1
    )
    assert correlations.shape == (4, 3, 3)
    assert np.allclose(correlations, np.swapaxes(correlations, 1, 2))


def test_uncoupled_atoms_decorrelate_in_the_trajectory_limit():
    """J = 0: independent dynamics, the connected correlation tends to 0."""
    couplings = np.zeros((2, 2))
    few = twa_correlations(
        couplings, TIMES_US, omega=12.0, delta=0.0, num_trajectories=64, dt_us=0.01, seed=0
    )
    many = twa_correlations(
        couplings, TIMES_US, omega=12.0, delta=0.0, num_trajectories=8000, dt_us=0.01, seed=0
    )
    off_few = np.abs(few[1:, 0, 1]).mean()
    off_many = np.abs(many[1:, 0, 1]).mean()
    assert off_many < off_few
    assert off_many < 5e-3


def test_coupled_atoms_build_correlation():
    """With strong coupling and a drive, the connected correlation becomes non-zero."""
    couplings = np.array([[0.0, 30.0], [30.0, 0.0]])
    correlations = twa_correlations(
        couplings, TIMES_US, omega=15.0, delta=5.0, num_trajectories=4000, dt_us=0.005, seed=2
    )
    assert np.abs(correlations[1:, 0, 1]).max() > 1e-3


def test_seed_is_reproducible():
    couplings = np.array([[0.0, 6.0], [6.0, 0.0]])
    first = twa_correlations(
        couplings, TIMES_US, omega=10.0, delta=1.0, num_trajectories=128, dt_us=0.01, seed=7
    )
    second = twa_correlations(
        couplings, TIMES_US, omega=10.0, delta=1.0, num_trajectories=128, dt_us=0.01, seed=7
    )
    assert np.array_equal(first, second)


def test_connected_helper_matches_definition():
    rng = np.random.default_rng(3)
    spins = rng.normal(size=(200, 3, 3))
    occupations = 0.5 * (1.0 - spins[..., 2])
    expected = np.cov(occupations.T, bias=True)
    assert np.allclose(_connected(spins), expected)
