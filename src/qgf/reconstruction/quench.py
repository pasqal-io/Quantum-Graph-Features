"""Exact Rydberg quench: Hamiltonian, connected correlations, two-atom reduction.

Two uses:

* the "exact two-atom correlation" physical control, which isolates the pair effect:
  whatever residual survives its inclusion is, by construction, beyond-pair physics;
* exact short-time data for the pipeline test (t <= 30 ns): the stored grid starts at
  100 ns, well past the perturbative regime, so the short-time test must be simulated.

Convention: hbar = 1, energies in rad/us, times in us, n_i = |1><1|_i, initial state
|0...0>. Returned correlations are in the occupation convention, in [-1/4, 1/4].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qgf.reconstruction.config import C6_RAD_UM6_PER_US, R_NN_UM, QuenchSpec

#: Beyond this, dense diagonalisation of 2^N becomes unreasonable.
MAX_EXACT_NODES: int = 14


def interaction_matrix(
    positions: np.ndarray,
    c6: float = C6_RAD_UM6_PER_US,
) -> np.ndarray:
    """Van der Waals interaction J_ij = C6 / d_ij^6 in rad/us, zero diagonal.

    The mask uses the actual distance: two coincident atoms stay at 0.0 instead of
    producing an inf.
    """
    coordinates = np.asarray(positions, dtype=float)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    positive = distances > 0
    couplings = np.zeros_like(distances)
    couplings[positive] = c6 / distances[positive] ** 6
    return couplings


def nearest_neighbour_coupling(r_nn: float = R_NN_UM, c6: float = C6_RAD_UM6_PER_US) -> float:
    """J_NN = C6 / r_NN^6 in rad/us, the reference energy scale of the quench."""
    return c6 / r_nn**6


def delta_infinity(couplings: np.ndarray) -> float:
    """delta_inf = sum_{i!=j} J_ij / (4 N), in rad/us."""
    num_nodes = couplings.shape[0]
    return float(couplings.sum() / (4.0 * num_nodes))


@dataclass(frozen=True)
class QuenchParameters:
    """Effective quench parameters for one graph."""

    omega: float
    delta: float
    couplings: np.ndarray

    @property
    def scaled_couplings(self) -> np.ndarray:
        """J in units of J_NN."""
        return self.couplings / nearest_neighbour_coupling()


def quench_parameters(
    positions: np.ndarray,
    quench: QuenchSpec,
    c6: float = C6_RAD_UM6_PER_US,
    r_nn: float = R_NN_UM,
) -> QuenchParameters:
    """Effective Omega and delta (rad/us) of a register, for a quench regime.

    AFM: Omega = 0.3 J_NN, delta = 0. FM: Omega = 0.5 J_NN, delta = 2 delta_inf, with
    delta_inf specific to each graph.
    """
    couplings = interaction_matrix(positions, c6=c6)
    j_nn = nearest_neighbour_coupling(r_nn=r_nn, c6=c6)
    return QuenchParameters(
        omega=quench.omega(j_nn),
        delta=quench.delta(couplings),
        couplings=couplings,
    )


# --------------------------------------------------------------------------------------
# Exact quench on 2^N states
# --------------------------------------------------------------------------------------


def occupation_table(num_nodes: int) -> np.ndarray:
    """(2^N, N) binary occupation table of every basis state."""
    states = np.arange(2**num_nodes, dtype=np.int64)
    return ((states[:, None] >> np.arange(num_nodes)[None, :]) & 1).astype(np.int8)


def rydberg_hamiltonian(couplings: np.ndarray, omega: float, delta: float) -> np.ndarray:
    """Dense (2^N, 2^N) H = sum_{i<j} J_ij n_i n_j + sum_i [Omega/2 sx_i - delta n_i]."""
    num_nodes = couplings.shape[0]
    if num_nodes > MAX_EXACT_NODES:
        raise ValueError(f"N={num_nodes} > {MAX_EXACT_NODES}: dense diagonalisation refused")
    dimension = 2**num_nodes
    bits = occupation_table(num_nodes)
    interaction = 0.5 * np.einsum("si,ij,sj->s", bits, couplings, bits)
    hamiltonian = np.zeros((dimension, dimension))
    np.fill_diagonal(hamiltonian, interaction - delta * bits.sum(axis=1))
    states = np.arange(dimension, dtype=np.int64)
    for node in range(num_nodes):
        flipped = states ^ (1 << node)
        hamiltonian[states, flipped] += 0.5 * omega
    return hamiltonian


def quench_correlations(
    couplings: np.ndarray,
    times_us: np.ndarray,
    omega: float,
    delta: float,
) -> np.ndarray:
    """Connected correlations C_ij(t) = <n_i n_j> - <n_i><n_j>, shape (N_t, N, N).

    Exact diagonalisation from |0...0>, valid for N <= MAX_EXACT_NODES.
    """
    num_nodes = couplings.shape[0]
    hamiltonian = rydberg_hamiltonian(couplings, omega, delta)
    energies, vectors = np.linalg.eigh(hamiltonian)
    # |psi(0)> = |0...0> = first basis vector.
    amplitudes = vectors[0, :]
    times = np.asarray(times_us, dtype=float)
    phases = np.exp(-1j * energies[None, :] * times[:, None])
    states = (phases * amplitudes[None, :]) @ vectors.T.astype(complex)
    probabilities = np.abs(states) ** 2
    bits = occupation_table(num_nodes).astype(float)
    means = probabilities @ bits
    joint = np.einsum("ts,si,sj->tij", probabilities, bits, bits)
    return joint - means[:, :, None] * means[:, None, :]


# --------------------------------------------------------------------------------------
# Exact two-atom reduction
# --------------------------------------------------------------------------------------
#: The two-atom quench from |00> stays in the symmetric subspace
#: {|00>, (|01>+|10>)/sqrt(2), |11>}: the antisymmetric sector is coupled neither by
#: Omega, nor by delta, nor by J. The problem is exactly 3x3.
_SYMMETRIC_DIMENSION: int = 3


def two_body_hamiltonians(couplings: np.ndarray, omega: float, delta: float) -> np.ndarray:
    """(P, 3, 3) Hamiltonians of the symmetric subspace, one per coupling J_ij."""
    values = np.asarray(couplings, dtype=float).reshape(-1)
    hopping = omega / np.sqrt(2.0)
    matrices = np.zeros((values.size, _SYMMETRIC_DIMENSION, _SYMMETRIC_DIMENSION))
    matrices[:, 0, 1] = hopping
    matrices[:, 1, 0] = hopping
    matrices[:, 1, 2] = hopping
    matrices[:, 2, 1] = hopping
    matrices[:, 1, 1] = -delta
    matrices[:, 2, 2] = values - 2.0 * delta
    return matrices


def two_body_correlations(
    couplings: np.ndarray,
    times_us: np.ndarray,
    omega: float,
    delta: float,
    chunk: int = 200_000,
) -> np.ndarray:
    """Connected correlation of the **two-atom** system, (P, N_t).

    Dimension-3 model (symmetric subspace) solved by batched diagonalisation.
    """
    values = np.asarray(couplings, dtype=float).reshape(-1)
    times = np.asarray(times_us, dtype=float)
    output = np.empty((values.size, times.size))
    for start in range(0, values.size, chunk):
        block = values[start : start + chunk]
        energies, vectors = np.linalg.eigh(two_body_hamiltonians(block, omega, delta))
        # <psi(0)| = |00> = first component of the symmetric basis.
        initial = vectors[:, 0, :]
        phases = np.exp(-1j * energies[:, None, :] * times[None, :, None])
        states = np.einsum("pts,pks->ptk", phases * initial[:, None, :], vectors)
        probabilities = np.abs(states) ** 2
        symmetric = probabilities[:, :, 1]
        doubly = probabilities[:, :, 2]
        mean = 0.5 * symmetric + doubly
        output[start : start + chunk] = doubly - mean**2
    return output
