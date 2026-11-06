"""Exact non-unit-disk benchmark (experiments/non_unit_disk): configs, analysis, released data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from qgf.downstream.gt.pe_transforms import CONFIGS, quantum_sources

ROOT = Path(__file__).resolve().parents[2]


def _script(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_configs_and_sources():
    assert quantum_sources("rrwp_lap_ring_geom_exact") == ["afm_exact", "fm_exact"]
    assert quantum_sources("rrwp_lap_ring_geom_exact_nn") == ["afm_exact_nn", "fm_exact_nn"]
    assert CONFIGS["rrwp_lap_ring_geom_exact"][2:] == CONFIGS["rrwp_lap_ring_geom"]


def _graph(n: int, gid: int, value: float) -> Data:
    src = torch.arange(n - 1)
    return Data(x=torch.eye(7)[torch.zeros(n, dtype=torch.long)],
                edge_index=torch.stack([src, src + 1]),
                edge_attr=torch.eye(4)[torch.zeros(n - 1, dtype=torch.long)],
                y=torch.tensor([gid % 2]), pos=torch.rand(n, 2), initial_id=torch.tensor(gid),
                onebody_features=torch.full((20, n), value),
                twobody_features=torch.full((20, n, n), value))  # fmt: skip


def test_subset_restricts_full_size_sources(tmp_path):
    """exact173: the 188-graph emulation files are cut to the subset before attaching."""
    from qgf.data.io import save_dataset
    from qgf.downstream.gt.data import load_mutag

    sizes = [10, 30, 24, 25, 12]
    meta = {"zz_scale_applied": 1.0}
    for name, v in [("afm", 1.0), ("fm", 2.0)]:
        save_dataset(tmp_path / f"mutag_{name}.pt",
                     [_graph(n, i, v) for i, n in enumerate(sizes)], meta)  # fmt: skip
    small = [(i, n) for i, n in enumerate(sizes) if n <= 24]
    save_dataset(tmp_path / "mutag_exact_afm.pt", [_graph(n, i, 3.0) for i, n in small], meta)
    graphs = load_mutag(["afm", "fm", "afm_exact"], tmp_path, subset="exact173")
    assert [int(g.initial_id) for g in graphs] == [0, 2, 4]
    assert all(float(g.onebody_fm[0, 0]) == 2.0 for g in graphs)
    assert all(float(g.twobody_afm_exact[0, 0, 1]) == 3.0 for g in graphs)
    assert len(load_mutag(["afm", "fm"], tmp_path)) == 5


def test_analyze_original_runs(tmp_path):
    ana = _script("experiments/non_unit_disk/analyze.py")
    ref = pd.read_csv(ROOT / "experiments/non_unit_disk/expected/trials.csv")
    assert len(ref) == 7 * 4 * 10 * 5
    assert set(ref.test_size + ref.val_size + ref.train_size) == {173}
    pairs = ana.paired(ref)
    assert len(pairs) == 5 and (pairs.n == 350).all()
    ana.main(["--out", str(tmp_path)])
    assert (tmp_path / "tables/paper/table_mutag_exact173.tex").exists()


def test_analyze_rerun_tree(tmp_path):
    """A re-run tree equal to the original trials of one backbone: zero deltas, trials kept."""
    ana = _script("experiments/non_unit_disk/analyze.py")
    ref = pd.read_csv(ROOT / "experiments/non_unit_disk/expected/trials.csv")
    for r in ref[ref.model == "GCN"].itertuples():
        d = tmp_path / "tree" / r.model / r.config / f"fold_{r.fold}" / f"seed_{r.seed}"
        d.mkdir(parents=True)
        res = {"model_type": r.model, "config": r.config, "fold": r.fold, "seed": r.seed,
               "test_f1_macro": r.value, **{k: getattr(r, k) for k in ana.EXTRA}}  # fmt: skip
        (d / "results.json").write_text(json.dumps(res))
    out = tmp_path / "out"
    ana.main(["--results", str(tmp_path / "tree"), "--out", str(out)])
    trials = pd.read_csv(out / "rerun_trials.csv")
    assert list(trials.columns) == list(ref.columns) and len(trials) == 200
    cmp = pd.read_csv(out / "rerun_vs_original.csv")
    assert len(cmp) == 4 and cmp.delta_over_std.max() < 1e-6
    assert (out / "tables/rerun/table_mutag_exact173.tex").exists()


EXACT = ["mutag_exact_afm", "mutag_exact_fm", "mutag_exact_nn_afm", "mutag_exact_nn_fm"]


@pytest.mark.slow
@pytest.mark.parametrize("name", EXACT)
def test_released_exact_files(name):
    """Shapes, conventions and graph alignment of the released exact files (needs the data)."""
    from qgf.data import DEFAULT_ROOT, file_name, load

    if not (Path(DEFAULT_ROOT) / file_name(name)).exists():
        pytest.skip("released exact datasets not available")
    graphs, meta = load(name)
    base, _ = load("mutag_afm")
    small = [g for g in base if g.num_nodes <= 24]
    assert len(graphs) == len(small) == 173 and meta["name"] == name
    assert meta["zz_scale_applied"] == 1.0
    np.testing.assert_allclose(meta["times_ns"], np.linspace(100, 4000, 20))
    for g, b in zip(graphs, small, strict=True):
        n = g.num_nodes
        assert int(g.initial_id) == int(b.initial_id) and n == b.num_nodes
        assert torch.equal(g.edge_index, b.edge_index) and torch.equal(g.y, b.y)
        assert g.occupation.shape == g.onebody_features.shape == (20, n)
        assert g.twobody_features.shape == (20, n, n)
        tb = g.twobody_features
        assert torch.all(tb[:, range(n), range(n)] == 0)
        torch.testing.assert_close(tb, tb.transpose(1, 2), atol=1e-6, rtol=0)
        occ = g.occupation.double()
        torch.testing.assert_close(g.onebody_features.double(), 4 * occ * (1 - occ),
                                   atol=1e-6, rtol=0)  # fmt: skip
        assert torch.all((occ >= -1e-6) & (occ <= 1 + 1e-6))
