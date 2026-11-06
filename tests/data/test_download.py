import hashlib
import json

import pytest

from qgf.data import download as dl
from qgf.data.manifest import write_manifest


@pytest.fixture
def record(tmp_path, monkeypatch):
    """A fake Zenodo record served from file:// URLs."""
    store = tmp_path / "store"
    store.mkdir()
    (store / "a.pt").write_bytes(b"alpha")
    (store / "b.pt").write_bytes(b"beta")
    (store / "qgf_raw.tar").write_bytes(b"raw")
    write_manifest(store, {"a.pt": 1, "b.pt": 2})
    files = [
        {
            "key": p.name,
            "checksum": "md5:" + hashlib.md5(p.read_bytes()).hexdigest(),
            "links": {"self": p.as_uri()},
        }
        for p in sorted(store.iterdir())
    ]
    api = tmp_path / "api"
    api.mkdir()
    (api / "123").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(dl, "API", api.as_uri() + "/")
    return store


def test_record_id_forms():
    for form in ("123", "10.5281/zenodo.123", "https://zenodo.org/records/123/"):
        assert dl.record_id(form) == "123"
    with pytest.raises(ValueError):
        dl.record_id("zenodo")


def test_unset_record_raises(monkeypatch):
    monkeypatch.setattr(dl, "ZENODO_RECORD", None)
    with pytest.raises(RuntimeError, match="No Zenodo record"):
        dl.download(None)


def test_download_default_and_subset(record, tmp_path):
    root = tmp_path / "out"
    paths = dl.download("10.5281/zenodo.123", root)
    assert sorted(p.name for p in paths) == ["a.pt", "b.pt", "manifest.json"]
    assert (root / "b.pt").read_bytes() == b"beta"
    assert not (root / "qgf_raw.tar").exists()

    assert [p.name for p in dl.download("123", root, files=["qgf_raw.tar"])] == ["qgf_raw.tar"]
    with pytest.raises(ValueError, match="not in the record"):
        dl.download("123", root, files=["nope.pt"])


def test_corrupted_source_is_rejected(record, tmp_path):
    (record / "a.pt").write_bytes(b"tampered")  # md5 in the record no longer matches
    with pytest.raises(OSError, match="md5"):
        dl.download("123", tmp_path / "out")
    assert not (tmp_path / "out" / "a.pt").exists()
