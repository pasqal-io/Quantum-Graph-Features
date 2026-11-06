"""Classical spin dynamics of the same quench (dTWA).

This is the **classical limit of the same Hamiltonian**: each spin precesses in the
effective field of the transverse field Omega, the longitudinal field delta and the
mean-field sum of the couplings sum_j J_ij n_j.

Strict mean field cancels connected correlations by construction. To get a non-trivial
correlation, the quantum fluctuations of the initial state are sampled: the **discrete
Truncated Wigner Approximation** (dTWA, Schachenmayer-Pikovski-Rey 2015). Each spin starts
at s^z = +1 (state |0>) with transverse components s^{x,y} drawn in {+-1}, reproducing
<(sigma^x)^2> = 1. Each trajectory then evolves deterministically and the connected
correlation is averaged over trajectories.
"""

from __future__ import annotations

import itertools

import numpy as np

#: Spins carry (s^x, s^y, s^z) on the last axis.
_X, _Y, _Z = 0, 1, 2


def _occupation(spins: np.ndarray) -> np.ndarray:
    """n_i = (1 - s^z_i)/2, shape (..., N)."""
    return 0.5 * (1.0 - spins[..., _Z])


def effective_field(
    spins: np.ndarray,
    couplings: np.ndarray,
    omega: float,
    delta: float,
) -> np.ndarray:
    """Mean-field effective field B_i, shape (..., N, 3).

    From H = sum_{i<j} J_ij n_i n_j + sum_i [(Omega/2) sigma^x_i - delta n_i] written in
    sigma: B^x = Omega/2, B^y = 0, B^z = delta/2 - (1/2) sum_j J_ij n_j.
    """
    occupations = _occupation(spins)  # (..., N)
    coupling_field = occupations @ couplings.T  # sum_j J_ij n_j (J symmetric)
    field_z = 0.5 * delta - 0.5 * coupling_field
    field = np.zeros_like(spins)
    field[..., _X] = 0.5 * omega
    field[..., _Z] = field_z
    return field


def _derivative(
    spins: np.ndarray,
    couplings: np.ndarray,
    omega: float,
    delta: float,
) -> np.ndarray:
    """ds_i/dt = 2 B_i x s_i (Bloch equation, classical limit of Heisenberg)."""
    field = effective_field(spins, couplings, omega, delta)
    return 2.0 * np.cross(field, spins)


def _rk4_step(
    spins: np.ndarray,
    dt: float,
    couplings: np.ndarray,
    omega: float,
    delta: float,
) -> np.ndarray:
    """One Runge-Kutta 4 step over all trajectories."""
    k1 = _derivative(spins, couplings, omega, delta)
    k2 = _derivative(spins + 0.5 * dt * k1, couplings, omega, delta)
    k3 = _derivative(spins + 0.5 * dt * k2, couplings, omega, delta)
    k4 = _derivative(spins + dt * k3, couplings, omega, delta)
    return spins + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initial_spins(
    num_nodes: int,
    num_trajectories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """dTWA initial conditions of |0...0>, shape (n_traj, N, 3).

    s^z = +1 (zero occupation), independent s^{x,y} in {+-1}: <s^{x,y}> = 0 with unit
    variance, reproducing the quantum fluctuation of the polarised coherent state.
    """
    spins = np.empty((num_trajectories, num_nodes, 3))
    spins[..., _X] = rng.choice((-1.0, 1.0), size=(num_trajectories, num_nodes))
    spins[..., _Y] = rng.choice((-1.0, 1.0), size=(num_trajectories, num_nodes))
    spins[..., _Z] = 1.0
    return spins


def twa_correlations(
    couplings: np.ndarray,
    times_us: np.ndarray,
    omega: float,
    delta: float,
    num_trajectories: int = 512,
    dt_us: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Classical connected correlations C_ij(t) = <n_i n_j> - <n_i><n_j>.

    Shape (N_t, N, N), occupation convention. The trajectory ensemble is integrated once
    by RK4 and sampled at ``times_us`` (assumed increasing and positive).
    """
    couplings = np.asarray(couplings, dtype=float)
    num_nodes = couplings.shape[0]
    times = np.asarray(times_us, dtype=float)
    rng = np.random.default_rng(seed)
    spins = initial_spins(num_nodes, num_trajectories, rng)

    # Integration grid: regular step dt with the target times inserted exactly.
    horizon = float(times.max())
    num_steps = max(int(np.ceil(horizon / dt_us)), 1)
    grid = np.linspace(0.0, horizon, num_steps + 1)
    grid = np.unique(np.concatenate([grid, times]))

    correlations = np.zeros((times.size, num_nodes, num_nodes))
    target_lookup = {float(t): index for index, t in enumerate(times)}
    if 0.0 in target_lookup:
        correlations[target_lookup[0.0]] = _connected(spins)

    for start, stop in itertools.pairwise(grid):
        spins = _rk4_step(spins, stop - start, couplings, omega, delta)
        index = target_lookup.get(float(stop))
        if index is not None:
            correlations[index] = _connected(spins)
    return correlations


def _connected(spins: np.ndarray) -> np.ndarray:
    """Connected (N, N) correlation averaged over trajectories (first axis)."""
    occupations = _occupation(spins)  # (n_traj, N)
    mean = occupations.mean(axis=0)  # (N,)
    joint = np.einsum("ti,tj->ij", occupations, occupations) / occupations.shape[0]
    return joint - np.outer(mean, mean)
