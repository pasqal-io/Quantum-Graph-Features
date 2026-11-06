"""Smoke tests of the experiment entry points on tiny inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from qgf.data import DEFAULT_ROOT
from qgf.data.io import save_dataset
from qgf.generate.raw import raw_path, save_raw

ROOT = Path(__file__).parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def _script(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_raw(path: Path, n_graphs: int = 2, times=(100, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    records = []
    for _ in range(n_graphs):
        rec = {}
        for t in times:
            shots = rng.integers(0, 2, size=(50, 4))
            keys, counts = np.unique(shots, axis=0, return_counts=True)
            rec[t] = {"".join(map(str, k)): int(c) for k, c in zip(keys, counts, strict=True)}
        records.append(rec)
    save_raw(path, records, list(range(n_graphs)), [4] * n_graphs, list(times))


def test_shot_budget_smoke(tmp_path):
    _tiny_raw(raw_path(tmp_path, "mutag_fm"))
    mod = _script("experiments/shot_budget/run.py")
    mod.main(
        ["--raw-dir", str(tmp_path), "--pulses", "fm", "--n-boot", "5", "--out", str(tmp_path)]
    )
    report = json.loads((tmp_path / "shot_budget.json").read_text())
    assert {r["N"]: r["shots"] for r in report["formula"]}[18] == 875
    assert len(report["empirical"]["fm"]["graphs"]) == 2


def test_layout_analyze_smoke(tmp_path):
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 5, 5))
    (tmp_path / "afm").mkdir()
    for mode, sigma, r, corr in [
        ("noise", 0.0, 0, ref),
        ("noise", 0.02, 0, ref + 0.01 * rng.normal(size=ref.shape)),
        ("reembed", 0.0, 0, rng.normal(size=ref.shape)),
    ]:
        np.savez(
            tmp_path / "afm" / f"g0_s{sigma:.3f}_r{r}_{mode}.npz",
            corr=corr,
            sigma_rel=sigma,
            graph_idx=0,
            mode=mode,
            realization=r,
        )
    mod = _script("experiments/layout_sensitivity/analyze.py")
    mod.main([str(tmp_path), "--out", str(tmp_path / "s.json")])
    s = json.loads((tmp_path / "s.json").read_text())["pooled_spearman"]
    assert s["sigma=0.02"]["mean"] > 0.9 > s["re-embedding"]["mean"]


def test_layout_noise_is_seeded_and_rescaled():
    pytest.importorskip("emu_sv")
    mod = _script("experiments/layout_sensitivity/run.py")
    pos = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    a, _ = mod.perturb_positions(pos, 0.3, np.random.default_rng(1), 5.0)
    b, _ = mod.perturb_positions(pos, 0.3, np.random.default_rng(1), 5.0)
    np.testing.assert_array_equal(a, b)
    assert mod.min_distance(b) >= 5.0 - 1e-9
    assert mod.select_graph_indices([3, 8, 9, 12, 20]) == [1, 2, 3, 4]


@pytest.mark.skipif(not raw_path(RAW, "zinc_embeddings_val").exists(), reason="raw data absent")
def test_zinc_subset_criterion_on_val(tmp_path):
    mod = _script("experiments/generation/zinc_subset.py")
    ids, pos, edges = mod.load_embeddings(raw_path(RAW, "zinc_embeddings_val"))
    labels = [mod.cluster(*mod.elu_gr(p, e)) for p, e in zip(pos, edges, strict=True)]
    assert len(ids) == 963 and sum(c not in mod.EXCLUDED for c in labels) == 946


@pytest.mark.skipif(
    not raw_path(RAW, "mutag_afm").exists() or not (DEFAULT_ROOT / "mutag_afm.pt").exists(),
    reason="raw or processed data absent",
)
def test_validate_raw_mutag():
    mod = _script("experiments/generation/validate_raw.py")
    r = mod.compare("mutag_afm", RAW)
    assert r["graphs"] == 188 and max(r["max_abs_diff"].values()) == 0.0


def test_generation_run_smoke(tmp_path):
    pytest.importorskip("emu_sv")
    pos = torch.tensor([[0.0, 0.0], [5.01, 0.0], [2.505, 4.4]])
    g = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        pos=pos,
        initial_id=torch.tensor([7]),
    )
    save_dataset(tmp_path / "in.pt", [g], {"name": "tiny"})
    mod = _script("experiments/generation/run.py")
    out = tmp_path / "tiny_afm"
    mod.main(
        [
            "--input",
            str(tmp_path / "in.pt"),
            "--pulse",
            "afm",
            "--shots",
            "50",
            "--cpu",
            "--out",
            str(out),
            "--features",
        ]
    )
    blob = torch.load(out.with_suffix(".pt"), weights_only=True)
    graph = blob["graphs"][0]
    assert graph["onebody_features"].shape == (20, 3)
    assert graph["twobody_features"].shape == (20, 3, 3)
    assert torch.all(graph["twobody_features"].diagonal(dim1=1, dim2=2) == 0)


def test_raw_path_layout(tmp_path):
    assert raw_path(tmp_path, "mutag_fm") == tmp_path / "emulation/mutag/fm.npz"
    assert raw_path(tmp_path, "zinc_afm_val") == tmp_path / "emulation/zinc/afm_val.npz"
    assert raw_path(tmp_path, "zinc_embeddings_test") == tmp_path / "embeddings/zinc_test.npz"
    assert raw_path(tmp_path, "mutag_qpu_afm_t4_715ns") == tmp_path / "qpu/mutag_afm/t4_715ns.npz"
