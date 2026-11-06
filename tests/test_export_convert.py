import enum
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

spec = importlib.util.spec_from_file_location(
    "export_internal_data",
    Path(__file__).parents[1] / "scripts" / "internal" / "export_internal_data.py",
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class State(enum.Enum):
    FREE = 1


def _internal_graph(n=3, t=4):
    g = Data()
    g.x = np.ones((n, 7))
    g.edge_index = torch.tensor([[0, 1], [1, 0]])
    g.y = [1]
    g.initial_ds_id = [42]
    g.embedded_position = [np.array([0.0, 0.0]), np.array([5.01, 0.0]), np.array([0.0, 5.01])]
    g.onebody_features = np.full((n, t), 0.25)  # [N, T] like residual datasets
    g.twobody_features = np.full((t, n, n), -0.1)
    g.embedding_state = State.FREE
    return g


def test_convert_scales_transposes_and_cleans():
    out = export.convert_graph(_internal_graph(), zz_scale=4.0, onebody_is_nt=True)

    assert out.onebody_features.shape == (4, 3)
    assert torch.allclose(out.onebody_features, torch.ones(4, 3))
    assert torch.allclose(out.twobody_features, torch.full((4, 3, 3), -0.4))
    assert out.onebody_features.dtype == torch.float32
    assert out.initial_id.tolist() == [42]
    assert "initial_ds_id" not in out and "embedding_state" not in out
    assert out.pos.shape == (3, 2) and "embedded_position" not in out
    assert out.y.tolist() == [1]


def test_convert_identity_scale_keeps_values():
    g = _internal_graph()
    g.onebody_features = np.full((4, 3), 1.0)  # already [T, N]
    out = export.convert_graph(g, zz_scale=1.0, onebody_is_nt=False)
    assert torch.allclose(out.onebody_features, torch.ones(4, 3))


def test_embedded_position_becomes_pos():
    out = export.convert_graph(_internal_graph(), zz_scale=1.0, onebody_is_nt=True)
    assert "embedded_position" not in out
    assert out.pos.shape == (3, 2) and out.pos.dtype == torch.float32


def test_pos_and_embedded_position_must_agree():
    g = _internal_graph()
    g.pos = torch.tensor([[0.0, 0.0], [5.01, 0.0], [0.0, 5.01]], dtype=torch.float64)
    out = export.convert_graph(g, zz_scale=1.0, onebody_is_nt=True)
    assert "embedded_position" not in out and out.pos.dtype == torch.float32

    g.pos = g.pos + 1.0
    with pytest.raises(AssertionError):
        export.convert_graph(g, zz_scale=1.0, onebody_is_nt=True)


def test_check_alignment():
    export.check_alignment({"a": [1, 2], "b": [1, 2]})
    with pytest.raises(ValueError, match="b"):
        export.check_alignment({"a": [1, 2], "b": [2, 1], "c": [1, 2]})


def test_kind_and_meta():
    assert export.kind("mutag_hw_afm") == "qpu"
    assert export.kind("mutag_afm_residual") == "residual"
    assert export.kind("zinc1500_fm") == "emulation"
    assert export.QPU_TIMES_NS == [305, 512, 715, 4000]
    assert "time_note" in export.make_meta("mutag_hw_afm", None, 4.0, [1])
    meta = export.make_meta("mutag_afm", "train", 1.0, [1])
    assert meta["kind"] == "emulation" and "source" not in meta and "time_note" not in meta


def test_occupation_source_is_converted_to_the_zz_convention():
    """ZINC_tiny sources store <n_i> as one-body and C (diagonal n(1-n)) as two-body."""
    g = _internal_graph()
    n = np.array([[0.2, 0.5, 0.9]] * 4)  # [T, N] occupations
    corr = np.full((4, 3, 3), -0.05)
    for i in range(3):
        corr[:, i, i] = n[:, i] * (1 - n[:, i])
    g.onebody_features, g.twobody_features = n, corr
    g.threebody_features = np.zeros((4, 3, 3, 3))
    out = export.convert_graph(g, zz_scale=4.0, onebody_is_nt=False, occupation=True)
    assert torch.allclose(out.occupation, torch.tensor(n, dtype=torch.float32))
    assert torch.allclose(out.onebody_features, torch.tensor(4 * n * (1 - n), dtype=torch.float32))
    assert torch.equal(out.twobody_features[:, 0, 0], torch.zeros(4))
    assert torch.allclose(out.twobody_features[:, 0, 1], torch.full((4,), -0.2))
    assert "threebody_features" not in out
    meta = export.make_meta("zinc_tiny_fm", "val", 4.0, None)
    assert meta["times_ns"] is None and "6000 ns" in meta["time_note"]
