"""Pulse sequences of the paper (App. D.1): constant AFM and FM quenches.

Device: a public ``pulser.devices.VirtualDevice`` built only from what the paper states:
Rydberg level 60 (``C6 / hbar = 8.66e5 um^6 rad/us``, pulser's table), nearest-neighbour
distance ``r_NN = 5.01 um``, a global Rydberg channel with maximum amplitude and detuning
``3 J_NN`` (``J_NN = C6 / r_NN^6``) and no modulation, and a large radial limit so that
every embedded register is accepted. Without modulation the emulated dynamics depend only
on ``C6``, the positions and the pulse (Omega, delta, duration); the channel limits are
validity checks.
"""

from functools import cache

import numpy as np
import pulser
from pulser.channels import Rydberg

PULSE_DURATION_NS = 4000
EVALUATION_TIMES_NS = [int(t) for t in np.linspace(100, PULSE_DURATION_NS, 20, dtype=int)]
AFM_OMEGA_OVER_JMAX = 0.3
FM_OMEGA_OVER_JMAX = 0.5
RYDBERG_LEVEL = 60
R_NN_UM = 5.01
MAX_RADIAL_DISTANCE_UM = 1000


def virtual_device(
    max_amp: float | None = None,
    max_abs_detuning: float | None = None,
    name: str = "QGF_virtual",
) -> pulser.devices.VirtualDevice:
    """Public device of the paper's emulations; limits default to ``3 J_NN``."""
    c6 = pulser.devices.VirtualDevice(
        name="probe", dimensions=2, rydberg_level=RYDBERG_LEVEL, channel_objects=()
    ).interaction_coeff
    j_nn = c6 / R_NN_UM**6
    channel = Rydberg.Global(
        max_abs_detuning=3 * j_nn if max_abs_detuning is None else max_abs_detuning,
        max_amp=3 * j_nn if max_amp is None else max_amp,
    )
    return pulser.devices.VirtualDevice(
        name=name,
        dimensions=2,
        rydberg_level=RYDBERG_LEVEL,
        min_atom_distance=R_NN_UM - 1e-3,  # registers have r_min = 5.01 um (float32)
        max_radial_distance=MAX_RADIAL_DISTANCE_UM,
        channel_objects=(channel,),
    )


@cache
def elevated_device() -> pulser.devices.VirtualDevice:
    """The emulation device (max amplitude/detuning ``3 J_NN``)."""
    return virtual_device()


def register_from_positions(pos) -> pulser.Register:
    return pulser.Register(
        {f"q{i}": (float(x), float(y)) for i, (x, y) in enumerate(np.asarray(pos))}
    )


def interactions(register: pulser.Register, c6: float) -> tuple[np.ndarray, float]:
    """``(J [N, N] with zero diagonal, J_max = C6 / r_min^6)``."""
    positions = np.array(list(register.qubits.values()))
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        j = np.where(dists != 0, c6 / dists**6, 0.0)
    return j, c6 / dists[dists > 0].min() ** 6


def graph_couplings(adjacency, c6: float) -> np.ndarray:
    """Graph Hamiltonian couplings ``J_NN A`` (``J_NN = C6 / r_NN^6``), zero diagonal.

    Not realisable by a register (no ``1 / r^6`` tails, any adjacency): emulated by
    passing it as the interaction matrix of the backend.
    """
    a = np.asarray(adjacency, dtype=float)
    j = (c6 / R_NN_UM**6) * a
    np.fill_diagonal(j, 0.0)
    return j


def quench_parameters(couplings: np.ndarray, pulse: str) -> tuple[float, float]:
    """``(Omega, delta)`` of a quench from a coupling matrix (``J_max = max J_ij``).

    For register couplings this equals :func:`afm_parameters` / :func:`fm_parameters`.
    """
    j = np.asarray(couplings, dtype=float)
    j_max = float(j.max())
    if pulse == "afm":
        return AFM_OMEGA_OVER_JMAX * j_max, 0.0
    if pulse == "fm":
        return FM_OMEGA_OVER_JMAX * j_max, float(j.sum()) / (2 * j.shape[0])
    raise ValueError(f"unknown pulse {pulse!r}")


def afm_parameters(register: pulser.Register, c6: float) -> tuple[float, float]:
    """AFM quench: ``(Omega, delta) = (0.3 J_max, 0)``."""
    _, j_max = interactions(register, c6)
    return AFM_OMEGA_OVER_JMAX * j_max, 0.0


def fm_parameters(register: pulser.Register, c6: float) -> tuple[float, float]:
    """FM quench: ``(Omega, delta) = (0.5 J_max, sum_{i != j} J_ij / (2N))``."""
    j, j_max = interactions(register, c6)
    return FM_OMEGA_OVER_JMAX * j_max, float(j.sum()) / (2 * len(register.qubits))


PARAMETERS = {"afm": afm_parameters, "fm": fm_parameters}


def quench_sequence(
    register: pulser.Register,
    pulse: str,
    device: pulser.devices.Device | None = None,
    duration: int = PULSE_DURATION_NS,
    couplings: np.ndarray | None = None,
) -> pulser.Sequence:
    """Constant global Rydberg pulse of ``duration`` ns for ``pulse`` in {"afm", "fm"}.

    ``couplings`` (e.g. :func:`graph_couplings`) sets the pulse from that matrix instead of
    the register's ``C6 / r^6``; emulate it with the same matrix as interaction matrix.
    """
    device = device or elevated_device()
    if couplings is None:
        omega, delta = PARAMETERS[pulse](register, device.interaction_coeff)
    else:
        omega, delta = quench_parameters(couplings, pulse)
    seq = pulser.Sequence(register=register, device=device)
    seq.declare_channel("rydberg", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(duration, omega, delta, 0), "rydberg")
    return seq
