"""Download the released QGF data from Zenodo.

    uv run python -m qgf.data.download [record] [--root data/processed] [--files a.pt b.pt]

``record`` is a Zenodo record id, record URL or DOI (``10.5281/zenodo.<id>``); it
defaults to :data:`ZENODO_RECORD`. Every file is checked against the md5 published by
Zenodo, and the processed datasets against ``manifest.json`` (sha256). Files already
present with the right checksum are skipped. The optional archives are only fetched
when named in ``files``: raw counts (``qgf_raw.tar``, ~1.4 GB; extract with
``tar xf data/processed/qgf_raw.tar -C data``) and the reference results of every
experiment (``qgf_results.tar``, a few MB; per-trial / per-cell JSON, CSV and parquet) and
the per-trial results of the bit-identical re-runs (``qgf_results_additions.tar``, ~40 MB).
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.request import urlopen

from qgf.data.manifest import MANIFEST, verify_manifest
from qgf.data.registry import DEFAULT_ROOT

ZENODO_RECORD: str | None = "23099924"  # DOI 10.5281/zenodo.23099924 (works once published)
API = "https://zenodo.org/api/records/"
RAW_ARCHIVE = "qgf_raw.tar"
RESULTS_ARCHIVE = "qgf_results.tar"


def record_id(record_url_or_doi: str) -> str:
    match = re.search(r"(\d+)/*$", str(record_url_or_doi).strip())
    if not match:
        raise ValueError(f"cannot read a Zenodo record id from {record_url_or_doi!r}")
    return match.group(1)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, md5: str | None) -> None:
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return
    part = dest.with_name(dest.name + ".part")
    with urlopen(url) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if md5 is not None and _md5(part) != md5:
        part.unlink()
        raise OSError(f"{dest.name}: md5 mismatch after download")
    part.replace(dest)


def download(
    record_url_or_doi: str | None = None,
    root: Path = DEFAULT_ROOT,
    files: list[str] | None = None,
) -> list[Path]:
    """Download ``files`` (default: manifest + all processed datasets) into ``root``."""
    record = record_url_or_doi or ZENODO_RECORD
    if record is None:
        raise RuntimeError(
            "No Zenodo record configured: pass the record id/URL/DOI "
            "(e.g. `python -m qgf.data.download 10.5281/zenodo.<id>`) "
            "or set qgf.data.download.ZENODO_RECORD."
        )
    with urlopen(API + record_id(record)) as response:
        entries = {f["key"]: f for f in json.load(response)["files"]}
    wanted = files or [k for k in entries if k == MANIFEST or k.endswith(".pt")]
    unknown = sorted(set(wanted) - set(entries))
    if unknown:
        raise ValueError(f"not in the record: {unknown}")
    if any(name.endswith(".pt") for name in wanted) and MANIFEST not in wanted:
        wanted = [MANIFEST, *wanted]

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in wanted:
        entry = entries[name]
        algo, _, digest = entry.get("checksum", "").partition(":")
        _fetch(entry["links"]["self"], root / name, digest if algo == "md5" else None)
        paths.append(root / name)

    if MANIFEST not in wanted:
        return paths
    problems = [p for p in verify_manifest(root) if p.split(":")[0] in wanted]
    if problems:
        raise OSError("manifest check failed: " + "; ".join(problems))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", nargs="?", help="Zenodo record id, URL or DOI")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--files", nargs="+", help=f"subset of files (e.g. {RAW_ARCHIVE}, {RESULTS_ARCHIVE})"
    )
    args = parser.parse_args()
    for path in download(args.record, args.root, args.files):
        print(path)


if __name__ == "__main__":
    main()
