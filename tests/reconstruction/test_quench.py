"""Exact Rydberg quench and its two-body reduction."""

from __future__ import annotations

import numpy as np
import pytest

from qgf.reconstruction.config import AFM_QUENCH, FM_QUENCH, R_NN_UM
from qgf.reconstruction.quench import (
    MAX_EXACT_NODES,
    delta_infinity,
    interaction_matrix,
    nearest_neighbour_coupling,
    occupation_table,
    quench_correlations,
    quench_parameters,
    rydberg_hamiltonian,
    two_body_correlations,
)

TIMES_US = np.array([0.0, 0.05, 0.1, 0.2])


def test_interaction_matrix_is_symmetric_with_zero_diagonal():
    positions = np.array([[0.0, 0.0], [R_NN_UM, 0.0], [0.0, 2 * R_NN_UM]])
    couplings = interaction_matrix(positions)
    assert np.allclose(couplings, couplings.T)
    assert np.allclose(np.diag(couplings), 0.0)
    assert couplings[0, 1] == pytest.approx(nearest_neighbour_coupling())


def test_interaction_matrix_guards_coincident_atoms():
    """Two coincident atoms stay at 0.0 instead of producing an inf."""
    couplings = interaction_matrix(np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert np.all(np.isfinite(couplings))
    assert couplings[0, 1] == 0.0


def test_delta_infinity_formula():
    couplings = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert delta_infinity(couplings) == pytest.approx(4.0 / (4 * 2))


def test_quench_parameters_follow_the_two_regimes():
    positions = np.array([[0.0, 0.0], [R_NN_UM, 0.0]])
    afm = quench_parameters(positions, AFM_QUENCH)
    fm = quench_parameters(positions, FM_QUENCH)
    assert afm.omega == pytest.approx(0.3 * nearest_neighbour_coupling())
    assert afm.delta == 0.0
    assert fm.omega == pytest.approx(0.5 * nearest_neighbour_coupling())
    assert fm.delta == pytest.approx(2.0 * delta_infinity(fm.couplings))
    assert afm.scaled_couplings[0, 1] == pytest.approx(1.0)


def test_occupation_table_enumerates_bitstrings():
    table = occupation_table(3)
    assert table.shape == (8, 3)
    assert table[0].tolist() == [0, 0, 0]
    assert table[7].tolist() == [1, 1, 1]
    assert table[1].tolist() == [1, 0, 0]


def test_hamiltonian_is_hermitian_with_expected_diagonal():
    couplings = np.array([[0.0, 3.0], [3.0, 0.0]])
    hamiltonian = rydberg_hamiltonian(couplings, omega=1.0, delta=0.5)
    assert np.allclose(hamiltonian, hamiltonian.T)
    # |00> -> 0 ; |10> et |01> -> -delta ; |11> -> J - 2 delta
    assert np.allclose(np.diag(hamiltonian), [0.0, -0.5, -0.5, 3.0 - 1.0])
    assert hamiltonian[0, 1] == pytest.approx(0.5)


def test_hamiltonian_refuses_large_systems():
    with pytest.raises(ValueError, match="refused"):
        rydberg_hamiltonian(np.zeros((MAX_EXACT_NODES + 1,) * 2), omega=1.0, delta=0.0)


def test_correlations_vanish_without_drive():
    """Without Omega, the state stays |0...0>: no connected correlation."""
    couplings = interaction_matrix(np.array([[0.0, 0.0], [R_NN_UM, 0.0], [0.0, R_NN_UM]]))
    correlations = quench_correlations(couplings, TIMES_US, omega=0.0, delta=0.0)
    assert np.allclose(correlations, 0.0)


def test_correlations_are_symmetric_and_bounded():
    """C_ij(t) = <n_i n_j> - <n_i><n_j> vit dans [-1/4, 1/4]."""
    positions = np.array([[0.0, 0.0], [R_NN_UM, 0.0], [2 * R_NN_UM, 0.0], [0.0, R_NN_UM]])
    parameters = quench_parameters(positions, AFM_QUENCH)
    correlations = quench_correlations(
        parameters.couplings,
        TIMES_US,
        parameters.omega,
        parameters.delta,
    )
    assert correlations.shape == (TIMES_US.size, 4, 4)
    assert np.allclose(correlations, np.swapaxes(correlations, 1, 2))
    assert np.allclose(correlations[0], 0.0)  # t = 0: product state
    assert correlations.min() >= -0.25 - 1e-12
    assert correlations.max() <= 0.25 + 1e-12


@pytest.mark.parametrize("quench", [AFM_QUENCH, FM_QUENCH])
def test_two_body_reduction_matches_full_two_atom_simulation(quench):
    """The symmetric 3x3 subspace exactly reproduces the 2-atom system."""
    positions = np.array([[0.0, 0.0], [1.3 * R_NN_UM, 0.0]])
    parameters = quench_parameters(positions, quench)
    times = np.array([0.01, 0.05, 0.15, 0.4])
    reference = quench_correlations(
        parameters.couplings,
        times,
        parameters.omega,
        parameters.delta,
    )[:, 0, 1]
    reduced = two_body_correlations(
        np.array([parameters.couplings[0, 1]]),
        times,
        parameters.omega,
        parameters.delta,
    )[0]
    assert np.allclose(reduced, reference, atol=1e-12)


def test_two_body_correlations_vectorise_over_pairs():
    couplings = np.array([0.0, 10.0, 40.0])
    values = two_body_correlations(couplings, np.array([0.05, 0.1]), omega=15.0, delta=0.0)
    assert values.shape == (3, 2)
    # Zero coupling: two independent atoms, zero connected correlation.
    assert np.allclose(values[0], 0.0)
    assert np.abs(values[1:]).max() > 1e-6


def test_two_body_chunking_is_transparent():
    couplings = np.linspace(0.0, 50.0, 37)
    times = np.array([0.05, 0.2])
    whole = two_body_correlations(couplings, times, omega=15.0, delta=1.0)
    chunked = two_body_correlations(couplings, times, omega=15.0, delta=1.0, chunk=5)
    assert np.allclose(whole, chunked)
