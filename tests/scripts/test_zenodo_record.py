import csv
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS / "zenodo"))
records = _module(SCRIPTS / "zenodo" / "build_record_files.py")
upload = _module(SCRIPTS / "zenodo_upload.py")


def test_build_record_files(tmp_path):
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    processed.mkdir()
    (raw / "layout_sensitivity").mkdir(parents=True)
    meta = {"name": "toy_hw", "split": None, "times_ns": [305], "kind": "qpu"}
    torch.save({"format_version": 1, "meta": meta, "graphs": []}, processed / "toy_hw.pt")
    (processed / "manifest.json").write_text(json.dumps({"toy_hw.pt": {"n_graphs": 3}}))
    raw_meta = json.dumps({"kind": "qpu", "name": "toy_counts"})
    np.savez(raw / "toy_counts.npz", initial_id=np.arange(2), meta=raw_meta)
    np.savez(raw / "layout_sensitivity" / "g0.npz", mode="noise")

    (tmp_path / "res.json").write_text("{}")
    with tarfile.open(tmp_path / "qgf_results.tar", "w") as tar:
        tar.add(tmp_path / "res.json", "results/x/res.json")

    readme, manifest = records.build(processed, raw, tmp_path / "out", tmp_path / "qgf_results.tar")
    rows = {r["filename"]: r for r in csv.DictReader(manifest.open())}
    assert set(rows) == {
        "toy_hw.pt",
        "raw/toy_counts.npz",
        "raw/layout_sensitivity/g0.npz",
        "results/x/res.json",
    }
    assert rows["results/x/res.json"]["sha256"] == records._digests(tmp_path / "res.json")[0]
    hw = rows["toy_hw.pt"]
    assert (hw["kind"], hw["n_graphs"]) == ("qpu", "3")
    assert hw["sha256"] == records._digests(processed / "toy_hw.pt")[0]
    assert rows["raw/toy_counts.npz"]["n_graphs"] == "2"
    text = readme.read_text()
    assert "`toy_hw.pt` | qpu | 3" in text and "1 layout-sensitivity" in text
    assert "$file_table" not in text and "$n_layout" not in text and "1 per-trial" in text


def test_already_uploaded_compares_md5(tmp_path):
    f = tmp_path / "a.pt"
    f.write_bytes(b"abc")
    assert upload.already_uploaded(f, {"a.pt": upload.md5(f)})
    assert not upload.already_uploaded(f, {"a.pt": "0" * 32})
    assert not upload.already_uploaded(f, {})


def _toy_release(tmp_path):
    processed, raw, results = tmp_path / "processed", tmp_path / "raw", tmp_path / "results"
    processed.mkdir()
    raw.mkdir()
    meta = {"name": "toy", "split": None, "times_ns": [100], "kind": "emulation"}
    torch.save({"format_version": 1, "meta": meta, "graphs": []}, processed / "toy.pt")
    (processed / "manifest.json").write_text(json.dumps({"toy.pt": {"n_graphs": 0}}))
    np.savez(raw / "toy.npz", initial_id=np.arange(1), meta=json.dumps(meta))
    trial = results / "downstream_gt" / "zinc" / "rrwp" / "seed_0"
    trial.mkdir(parents=True)
    (trial / "results.json").write_text("{}")
    (trial / "train.log").write_text("not shipped")
    return processed, raw, results


def test_results_members_filters_and_renames(tmp_path):
    _, _, results = _toy_release(tmp_path)
    (results / "intrinsic" / "corrected" / "MUTAG" / "RRWP" / "t").mkdir(parents=True)
    (results / "intrinsic" / "corrected" / "MUTAG" / "RRWP" / "t" / "result.json").write_text("{}")
    names = [a for _, a in upload.results_members(results)]
    assert names == [
        "results/downstream_gt/zinc/rrwp/seed_0/results.json",
        "results/intrinsic/grid/MUTAG/RRWP/t/result.json",
    ]


def test_replace_deletes_only_named_managed_files(tmp_path, monkeypatch):
    processed, raw, results = _toy_release(tmp_path)
    draft = {
        "id": 7,
        "links": {"bucket": "https://z/bucket", "html": "https://z/7"},
        "files": [
            {"filename": n, "checksum": "0", "links": {"self": f"https://z/files/{n}"}}
            for n in ("qgf_raw.tar", "README.md", "other.pdf")
        ],
    }
    calls = []

    def fake_api(base, token, method, path, body=None, size=None):
        calls.append((method, path))
        if method == "DELETE":
            draft["files"] = [f for f in draft["files"] if f["links"]["self"] != path]
        return draft if method == "GET" else {}

    monkeypatch.setattr(upload, "api", fake_api)
    monkeypatch.setenv("ZENODO_TOKEN", "t")
    argv = ["x", "--processed", str(processed), "--raw", str(raw), "--results", str(results)]
    argv += ["--deposition-id", "7", "--replace", "qgf_raw.tar", "README.md"]
    monkeypatch.setattr(sys, "argv", argv)
    upload.main()
    deleted = [p for m, p in calls if m == "DELETE"]
    assert deleted == ["https://z/files/qgf_raw.tar", "https://z/files/README.md"]
    assert not any(m == "PUT" and p == "deposit/depositions/7" for m, p in calls)  # no metadata
    uploaded = {p.rsplit("/", 1)[1] for m, p in calls if m == "PUT"}
    assert {"qgf_raw.tar", "qgf_results.tar", "README.md", "MANIFEST.csv"} <= uploaded
    with tarfile.open(processed / "qgf_results.tar") as tar:
        assert "results/README.md" in tar.getnames()
        assert not any(n.endswith(".log") for n in tar.getnames())
    for archive in ("qgf_raw.tar", "qgf_results.tar"):
        with tarfile.open(processed / archive) as tar:
            assert {(m.uid, m.gid, m.uname, m.gname) for m in tar} == {(0, 0, "", "")}

    monkeypatch.setattr(sys, "argv", [*argv[:-2], "other.pdf"])
    with pytest.raises(SystemExit, match="only accepts"):
        upload.main()
