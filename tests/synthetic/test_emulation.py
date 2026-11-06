"""The ladder quench through the public pulser device and emu-sv equals an independent
sparse-Krylov propagation of the paper Hamiltonian (6-atom ladder)."""

import numpy as np
import pytest

pytest.importorskip("emu_sv")

from qgf.generate.emulate import (  # noqa: E402 (after importorskip)
    DT_NS,
    MPS_MAX_BOND_DIM,
    emulate_expectations,
    make_config,
)
from qgf.generate.sequences import (  # noqa: E402
    elevated_device,
    quench_sequence,
    register_from_positions,
)
from qgf.synthetic.dataset import TIMES_NS  # noqa: E402
from qgf.synthetic.ladder import ladder, unit_disk_positions  # noqa: E402


def krylov_quench(pos: np.ndarray, pulse: str, times_ns: np.ndarray, c6: float):
    """Exact ``(occ [T, n], connected [T, n, n])`` of
    ``H = sum_{i<j} C6/r^6 n_i n_j + sum_i (Omega/2) X_i - delta n_i`` from |0...0>."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    n = len(pos)
    r = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    U = np.where(r > 0, c6 / np.where(r > 0, r, 1.0) ** 6, 0.0)
    j_max = c6 / r[r > 0].min() ** 6
    omega, delta = (0.3 * j_max, 0.0) if pulse == "afm" else (0.5 * j_max, U.sum() / (2 * n))
    dim = 2**n
    states = np.arange(dim)
    bits = ((states[None, :] >> np.arange(n)[::-1][:, None]) & 1).astype(float)
    diag = np.einsum("ij,ia,ja->a", np.triu(U, 1), bits, bits) - delta * bits.sum(0)
    rows = [states] + [states] * n
    cols = [states] + [states ^ (1 << (n - 1 - k)) for k in range(n)]
    vals = [diag.astype(complex)] + [np.full(dim, omega / 2, dtype=complex)] * n
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))))
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    t_us = np.asarray(times_ns) / 1000
    psis = expm_multiply(-1j * H, psi0, start=t_us[0], stop=t_us[-1], num=len(t_us))
    occ, conn = [], []
    for psi in psis:
        p = np.abs(psi) ** 2
        o = bits @ p
        occ.append(o)
        conn.append((bits * p) @ bits.T - np.outer(o, o))
    return np.array(occ), np.array(conn)


@pytest.mark.filterwarnings("ignore:.*initialized without any observables")
def test_make_config_defaults_are_the_paper_settings():
    _, sv = make_config(4, [], use_gpu=False)
    assert sv.dt == DT_NS == 10
    _, mps = make_config(30, [], use_gpu=False)
    assert mps.max_bond_dim == MPS_MAX_BOND_DIM == 1024
    _, mps16 = make_config(4, [], use_gpu=False, dt=20, sv_max_qubits=0, max_bond_dim=16)
    assert mps16.dt == 20 and mps16.max_bond_dim == 16


@pytest.mark.parametrize("pulse", ["afm", "fm"])
def test_emu_sv_ladder_quench_matches_krylov(pulse):
    pos = unit_disk_positions(ladder(3, (1,)))
    seq = quench_sequence(register_from_positions(pos), pulse)
    occ, conn = emulate_expectations(seq, TIMES_NS, use_gpu=False)
    occ_k, conn_k = krylov_quench(pos, pulse, TIMES_NS, elevated_device().interaction_coeff)
    assert occ.shape == (20, 6) and conn.shape == (20, 6, 6)
    assert np.abs(occ - occ_k).max() < 1e-3
    assert np.abs(conn - conn_k).max() < 1e-3


def test_ladder_pulse_parameters():
    reg = register_from_positions(unit_disk_positions(ladder(11, (2, 5))))
    afm = quench_sequence(reg, "afm")._schedule["rydberg"][-1].type
    assert afm.amplitude[0] == pytest.approx(16.42, abs=0.01)  # 0.3 J_max at 5.01 um
