"""Exact non-unit-disk features (experiments/non_unit_disk): couplings, pulses, emulation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

pulser = pytest.importorskip("pulser")

from qgf.generate.sequences import (  # noqa: E402
    R_NN_UM,
    afm_parameters,
    elevated_device,
    fm_parameters,
    graph_couplings,
    interactions,
    quench_parameters,
    quench_sequence,
    register_from_positions,
)

ROOT = Path(__file__).resolve().parents[2]


def _script(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _square() -> tuple[np.ndarray, np.ndarray]:
    """4-cycle register at r_NN (diagonal 1.41 r_NN) and its adjacency."""
    pos = R_NN_UM * np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    a = np.zeros((4, 4))
    for i in range(4):
        a[i, (i + 1) % 4] = a[(i + 1) % 4, i] = 1.0
    return pos, a


def test_quench_parameters_from_register_couplings_match_the_register_rule():
    pos, _ = _square()
    reg, c6 = register_from_positions(pos), elevated_device().interaction_coeff
    j, _ = interactions(reg, c6)
    assert quench_parameters(j, "afm") == pytest.approx(afm_parameters(reg, c6))
    assert quench_parameters(j, "fm") == pytest.approx(fm_parameters(reg, c6))


def test_graph_couplings_drop_the_tails():
    pos, a = _square()
    c6 = elevated_device().interaction_coeff
    j = graph_couplings(a, c6)
    j_nn = c6 / R_NN_UM**6
    np.testing.assert_allclose(j, j_nn * a)
    omega, delta = quench_parameters(j, "fm")
    assert omega == pytest.approx(0.5 * j_nn)
    assert delta == pytest.approx(j_nn * 8 / 8)  # sum_{i != j} J_ij / (2N) = J_NN |E| / N
    seq = quench_sequence(register_from_positions(pos), "fm", couplings=j)
    assert seq.get_duration() == 4000


def test_emulated_graph_hamiltonian_matches_exact_diagonalisation():
    """emu-sv with the J_NN A interaction matrix == dense ED of the same Hamiltonian."""
    pytest.importorskip("emu_sv")
    from qgf.reconstruction.quench import quench_correlations

    gen = _script("experiments/non_unit_disk/generate.py")
    pos, a = _square()
    g = Data(pos=torch.tensor(pos), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
             num_nodes=4)  # fmt: skip
    j = gen.couplings_of(g, "nn")
    np.testing.assert_allclose(j, graph_couplings(a, elevated_device().interaction_coeff))
    _, ob, tb = gen.exact_features(g, "nn", "afm", use_gpu=False)
    omega, delta = quench_parameters(j, "afm")
    c = quench_correlations(j, gen.TIMES_NS / 1000.0, omega, delta)
    off = ~np.eye(4, dtype=bool)
    assert np.abs(tb.numpy()[:, off] - 4 * c[:, off]).max() < 2e-3  # dt = 10 ns steps
    assert ob.shape == (20, 4) and torch.all(tb[:, range(4), range(4)] == 0)
