import json

from qgf.data.manifest import sha256, verify_manifest, write_manifest


def test_write_and_verify(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"hello")
    write_manifest(tmp_path, {"a.pt": 3})

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["a.pt"] == {"sha256": sha256(tmp_path / "a.pt"), "n_graphs": 3}
    assert verify_manifest(tmp_path) == []


def test_detects_corruption_and_missing(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"hello")
    (tmp_path / "b.pt").write_bytes(b"world")
    write_manifest(tmp_path, {"a.pt": 1, "b.pt": 1})
    (tmp_path / "a.pt").write_bytes(b"tampered")
    (tmp_path / "b.pt").unlink()

    problems = verify_manifest(tmp_path)

    assert any("a.pt" in p and "checksum" in p for p in problems)
    assert any("b.pt" in p and "missing" in p for p in problems)


def test_write_manifest_merges_existing_entries(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "b.pt").write_bytes(b"b")
    write_manifest(tmp_path, {"a.pt": 3, "b.pt": 2})
    (tmp_path / "a.pt").write_bytes(b"a2")  # re-exported
    write_manifest(tmp_path, {"a.pt": 4})
    assert verify_manifest(tmp_path) == []
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["a.pt"]["n_graphs"] == 4 and m["b.pt"]["n_graphs"] == 2
