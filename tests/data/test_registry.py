from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from qgf.data import file_name, load, load_splits
from qgf.data.io import save_dataset
from qgf.data.manifest import write_manifest


def test_file_names():
    assert file_name("mutag_afm") == "mutag_afm.pt"
    assert file_name("zinc_fm", "val") == "zinc_fm_val.pt"
    assert file_name("mutag_hw_afm") == "mutag_hw_afm.pt"
    assert file_name("ladder_afm") == "ladder_afm.pt"


@pytest.mark.parametrize(
    "name, split",
    [("unknown", None), ("mutag_afm", "train"), ("zinc_fm", None), ("zinc_fm", "dev")],
)
def test_file_name_errors(name, split):
    with pytest.raises(ValueError):
        file_name(name, split)


def test_load_and_load_splits(tmp_path):
    g = Data(x=torch.zeros(2, 1))
    save_dataset(tmp_path / "mutag_afm.pt", [g], {"name": "mutag_afm"})
    for split, n in [("train", 3), ("val", 1), ("test", 2)]:
        save_dataset(tmp_path / f"zinc_fm_{split}.pt", [g] * n, {"name": "zinc_fm"})

    graphs, meta = load("mutag_afm", root=tmp_path)
    splits, split_meta = load_splits("zinc_fm", root=tmp_path)

    assert len(graphs) == 1 and meta["name"] == "mutag_afm"
    assert {k: len(v) for k, v in splits.items()} == {"train": 3, "val": 1, "test": 2}
    assert split_meta["name"] == "zinc_fm"


def test_verify_uses_default_root(tmp_path, monkeypatch):
    import importlib

    from qgf.data import registry

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QGF_DATA", raising=False)  # default root, whatever the caller's env
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "a.pt").write_bytes(b"test")
    write_manifest(tmp_path / "data" / "processed", {"a.pt": 1})
    try:
        assert importlib.reload(registry).verify() == []
    finally:
        monkeypatch.undo()
        importlib.reload(registry)


def test_default_root_honours_env(monkeypatch):
    import importlib

    from qgf.data import registry

    monkeypatch.setenv("QGF_DATA", "/somewhere")
    try:
        assert Path("/somewhere") == importlib.reload(registry).DEFAULT_ROOT
    finally:
        monkeypatch.undo()
        importlib.reload(registry)
