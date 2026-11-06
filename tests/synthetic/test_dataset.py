import numpy as np
import pytest
import torch

from qgf.data.io import load_dataset, save_dataset
from qgf.synthetic.dataset import benchmark_inputs, ladder_data, make_meta, quench_arrays
from qgf.synthetic.ladder import ladder, unit_disk_positions


def _quench(n, T=20, seed=0):
    rng = np.random.default_rng(seed)
    occ = rng.uniform(0.05, 0.6, size=(T, n))
    conn = rng.normal(scale=0.05, size=(T, n, n))
    conn = conn + conn.transpose(0, 2, 1)
    conn[:, np.arange(n), np.arange(n)] = occ * (1 - occ)
    return occ, conn


def _graph(m, cr, i, seed=0, folds=(0, 1, 2)):
    occ, conn = _quench(2 * m, seed=seed)
    return (
        ladder_data(
            m,
            cr,
            y=i % 2,
            degeneracy=1 + i % 2,
            initial_id=i,
            blind_group=i // 2,
            cv_fold=folds,
            occupation=occ,
            connected=conn,
        ),
        occ,
        conn,
    )


def test_ladder_data_follows_the_qgf_conventions():
    g, occ, conn = _graph(4, (1,), 3)
    n = 8
    assert g.edge_index.shape == (2, 2 * (3 * 4 - 2 + 1))
    assert torch.equal(g.pos, torch.from_numpy(unit_disk_positions(ladder(4, (1,)))))
    assert g.onebody_features.shape == (20, n) and g.twobody_features.shape == (20, n, n)
    np.testing.assert_array_equal(g.onebody_features.numpy(), 4 * occ * (1 - occ))  # 4 C_ii
    assert torch.all(g.twobody_features[:, torch.arange(n), torch.arange(n)] == 0)
    occ_back, C_back = quench_arrays(g, 4.0)
    off = ~np.eye(n, dtype=bool)
    np.testing.assert_array_equal(C_back[:, off], conn[:, off])  # x4 / 4 is exact
    np.testing.assert_array_equal(occ_back, occ)
    assert g.cv_fold.tolist() == [0, 1, 2] and int(g.blind_group) == 1


def test_shape_validation():
    occ, conn = _quench(8)
    with pytest.raises(ValueError):
        ladder_data(4, (1,), y=1, degeneracy=1, initial_id=0, occupation=occ.T, connected=conn)


def test_round_trip_and_alignment(tmp_path):
    graphs = [_graph(4, (c,), i, seed=i)[0] for i, c in enumerate([0, 1, 2, 0])]
    save_dataset(tmp_path / "a.pt", graphs, make_meta("afm", "test"))
    loaded = load_dataset(tmp_path / "a.pt")
    assert loaded[1]["zz_scale_applied"] == 4.0
    assert loaded[0][0].twobody_features.dtype == torch.float64
    inputs = benchmark_inputs(loaded, loaded)
    assert inputs["F"]["afm"].shape == (4, 600) and inputs["folds"].shape == (4, 3)
    assert inputs["y"].tolist() == [0, 1, 0, 1]
    shuffled = (loaded[0][::-1], loaded[1])
    with pytest.raises(ValueError, match="misaligned"):
        benchmark_inputs(loaded, shuffled)
