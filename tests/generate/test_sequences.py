import numpy as np
import pytest

pulser = pytest.importorskip("pulser")

from qgf.generate.qpu import (  # noqa: E402 (after importorskip)
    afm_qpu_sequence,
    qpu_pulse_duration,
    register_rescale_factor,
    shots_clt,
)
from qgf.generate.sequences import (  # noqa: E402 (after importorskip)
    elevated_device,
    quench_sequence,
    register_from_positions,
    virtual_device,
)

TRIANGLE = [(0.0, 0.0), (5.01, 0.0), (2.505, 4.4)]


def _pulse(seq):
    return seq._schedule["rydberg"][-1].type


def test_device_is_public_with_raised_limits():
    dev = elevated_device()
    assert isinstance(dev, pulser.devices.VirtualDevice)
    assert dev.rydberg_level == 60
    assert dev.interaction_coeff == pytest.approx(8.66e5, rel=1e-3)  # C6 quoted in the paper
    j_nn = dev.interaction_coeff / 5.01**6
    ch = dev.channel_objects[0]
    assert ch.max_amp == pytest.approx(3 * j_nn) and ch.max_abs_detuning == pytest.approx(3 * j_nn)
    assert ch.mod_bandwidth is None  # emulation without modulation


def test_afm_and_fm_parameters():
    reg = register_from_positions(TRIANGLE)
    c6 = elevated_device().interaction_coeff
    pos = np.array(TRIANGLE)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    j = np.where(d > 0, c6 / np.where(d > 0, d, 1) ** 6, 0)
    j_max = c6 / 5.01**6
    afm = _pulse(quench_sequence(reg, "afm"))
    fm = _pulse(quench_sequence(reg, "fm"))
    assert afm.duration == 4000
    assert afm.amplitude[0] == pytest.approx(0.3 * j_max) and afm.detuning[0] == 0
    assert fm.amplitude[0] == pytest.approx(0.5 * j_max)
    assert fm.detuning[0] == pytest.approx(j.sum() / 6)


def test_shot_budget():
    # 66.67 * k_18^2 = 874.57: the paper rounds to 874, the code (and the QPU runs) use ceil.
    assert shots_clt(18) == 875
    assert shots_clt(10) == 734 and shots_clt(28) == 983  # MUTAG range used on the QPU
    with pytest.raises(ValueError):
        shots_clt(0)


def test_qpu_sequence_rescaled_time_corrected_and_padded():
    reg = register_from_positions(TRIANGLE)
    dev = virtual_device(max_amp=10.0, name="low_amp")  # stands for a hardware amplitude cap
    scale = register_rescale_factor(reg, dev)
    assert scale > 1  # 0.3 J_max at 5.01 um exceeds the amplitude cap
    seq = afm_qpu_sequence(reg, dev, 715)
    assert seq.get_duration() == 6000
    pulse = _pulse(seq)
    assert pulse.duration == qpu_pulse_duration(715, scale) and pulse.duration % 4 == 0
    assert pulse.amplitude[0] == pytest.approx(0.99 * dev.channel_objects[0].max_amp, rel=1e-6)
