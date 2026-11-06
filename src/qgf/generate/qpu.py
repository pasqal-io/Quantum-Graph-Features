"""Hardware run helpers (App. D.3, D.4): shot budget, register rescaling, padded AFM sequence.

``shots_clt`` and ``qpu_pulse_duration`` are pure (no pulser); the sequence helpers import
pulser lazily so that core-only installs can use the former.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    import pulser

TOTAL_SEQUENCE_DURATION_NS = 6000  # hardware maximum, every job is padded to it
CLOCK_PERIOD_NS = 4


def shots_clt(n: int, v_max: float = 0.06, eps: float = 0.03, delta: float = 0.05) -> int:
    """Shots so that all ``M = N(N+1)/2`` connected correlations are within ``eps``
    simultaneously with probability ``1 - delta`` (CLT + Bonferroni, Eq. 23-24)."""
    if n <= 0:
        raise ValueError("N must be a positive integer")
    if v_max <= 0 or eps <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("need v_max > 0, eps > 0 and 0 < delta < 1")
    k = norm.ppf(1.0 - delta / (n * (n + 1)))
    return math.ceil(v_max / eps**2 * k**2)


def register_rescale_factor(
    register: pulser.Register, device: pulser.devices.Device, safety: float = 0.99
) -> float:
    """Factor ``s >= 1`` on all positions so that ``0.3 J_max`` = ``safety * max_amp``:
    ``s = (0.3 C6 / (r_min^6 safety max_amp))^(1/6)`` (1 if the register already fits)."""
    from qgf.generate.sequences import AFM_OMEGA_OVER_JMAX, interactions

    _, j_max = interactions(register, device.interaction_coeff)
    target = safety * device.channel_objects[0].max_amp
    return max((AFM_OMEGA_OVER_JMAX * j_max / target) ** (1.0 / 6.0), 1.0)


def qpu_pulse_duration(t_emu_ns: int, scale: float) -> int:
    """Scaling positions by ``s`` divides ``J`` (and Omega) by ``s^6``; the pulse must last
    ``t_emu * s^6`` to reach the same state, floored to the 4 ns clock period."""
    return int(t_emu_ns * scale**6) // CLOCK_PERIOD_NS * CLOCK_PERIOD_NS


def afm_qpu_sequence(
    register: pulser.Register,
    device: pulser.devices.Device,
    t_emu_ns: int,
    total_ns: int = TOTAL_SEQUENCE_DURATION_NS,
) -> pulser.Sequence:
    """Rescaled, time-corrected AFM quench preceded by a delay up to ``total_ns``."""
    import pulser

    from qgf.generate.sequences import AFM_OMEGA_OVER_JMAX, interactions

    scale = register_rescale_factor(register, device)
    if scale > 1.0:
        register = pulser.Register({q: np.array(p) * scale for q, p in register.qubits.items()})
    if isinstance(device, pulser.devices.Device):  # real QPUs need a trap layout
        register = register.with_automatic_layout(device)
    _, j_max = interactions(register, device.interaction_coeff)
    duration = qpu_pulse_duration(t_emu_ns, scale)
    if scale > 1.0 and duration <= t_emu_ns:
        raise AssertionError(f"t_QPU={duration} <= t_emu={t_emu_ns} (scale={scale:.6f})")
    pad = total_ns - duration
    if pad < 0:
        raise ValueError(f"pulse of {duration} ns exceeds the {total_ns} ns sequence")
    seq = pulser.Sequence(register=register, device=device)
    seq.declare_channel("rydberg", "rydberg_global")
    if pad > 0:
        seq.delay(pad, "rydberg")
    seq.add(pulser.Pulse.ConstantPulse(duration, AFM_OMEGA_OVER_JMAX * j_max, 0.0, 0), "rydberg")
    return seq
