"""Build the Zenodo record ``README.md`` and ``MANIFEST.csv`` from the released data.

    uv run python scripts/zenodo/build_record_files.py [--processed data/processed]
        [--raw data/raw] [--results-tar data/processed/qgf_results.tar] [--out data/processed]

Reads ``<processed>/manifest.json`` (processed datasets), every file under ``<raw>``
(shipped inside ``qgf_raw.tar`` as ``raw/...``) and the members of ``qgf_results.tar``
(``results/...``, if it exists), and renders ``README.template.md``.
"""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from string import Template

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ["filename", "size", "sha256", "md5", "n_graphs", "kind", "description"]


def _digests(path_or_file) -> tuple[str, str]:
    """sha256 and md5 of a path or of an open binary file object."""
    if isinstance(path_or_file, Path):
        with open(path_or_file, "rb") as f:
            return _digests(f)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    for chunk in iter(lambda: path_or_file.read(1 << 20), b""):
        sha.update(chunk)
        md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def _results_rows(results_tar: Path) -> list[dict]:
    rows = []
    with tarfile.open(results_tar) as tar:
        for m in tar.getmembers():
            if m.isfile():
                sha, md5 = _digests(tar.extractfile(m))
                desc = "result file (see results/README.md inside the archive)"
                rows.append(
                    {
                        "filename": m.name,
                        "size": m.size,
                        "sha256": sha,
                        "md5": md5,
                        "n_graphs": "",
                        "kind": "result",
                        "description": desc,
                    }
                )
    return rows


def _processed_row(path: Path, n_graphs: int) -> dict:
    meta = torch.load(path, weights_only=True)["meta"]
    split = f", {meta['split']} split" if meta.get("split") else ""
    desc = f"{meta['kind']} features {meta['name']}{split}, times_ns={meta['times_ns']}"
    return {"n_graphs": n_graphs, "kind": meta["kind"], "description": desc}


def _raw_row(path: Path) -> dict:
    with np.load(path) as z:
        if "meta" in z.files:  # bitstring counts (qgf.generate.raw format)
            meta = json.loads(str(z["meta"]))
            desc = f"{meta['kind']} bitstring counts {meta['name']}"
            return {"n_graphs": len(z["initial_id"]), "kind": "counts", "description": desc}
        if "pos" in z.files:
            n = len(z["initial_id"])
            return {"n_graphs": n, "kind": "register", "description": "atom positions (um)"}
        return {"n_graphs": 1, "kind": "layout_sensitivity", "description": str(z["mode"])}


def manifest_rows(processed: Path, raw: Path, results_tar: Path | None = None) -> list[dict]:
    rows = []
    for name, entry in sorted(json.loads((processed / "manifest.json").read_text()).items()):
        path = processed / name
        rows.append({"filename": name, "path": path, **_processed_row(path, entry["n_graphs"])})
    for path in sorted(p for p in raw.rglob("*") if p.is_file()) if raw.exists() else []:
        rows.append({"filename": f"raw/{path.relative_to(raw)}", "path": path, **_raw_row(path)})
    for row in rows:
        path = row.pop("path")
        row["size"] = path.stat().st_size
        row["sha256"], row["md5"] = _digests(path)
    if results_tar is not None and results_tar.exists():
        rows += _results_rows(results_tar)
    return rows


def render_readme(rows: list[dict]) -> str:
    table = "\n".join(
        f"| `{r['filename']}` | {r['kind']} | {r['n_graphs']} | {r['size'] / 1e6:.1f} |"
        for r in rows
        if not r["filename"].startswith(("raw/layout_sensitivity/", "results/"))
    )
    n_layout = sum(r["filename"].startswith("raw/layout_sensitivity/") for r in rows)
    n_results = sum(r["filename"].startswith("results/") for r in rows)
    template = Template((HERE / "README.template.md").read_text())
    return template.substitute(file_table=table, n_layout=n_layout, n_results=n_results)


def build(processed: Path, raw: Path, out: Path, results_tar: Path | None = None) -> list[Path]:
    rows = manifest_rows(processed, raw, results_tar)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "MANIFEST.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "README.md").write_text(render_readme(rows))
    return [out / "README.md", out / "MANIFEST.csv"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument(
        "--results-tar", type=Path, default=ROOT / "data" / "processed" / "qgf_results.tar"
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = p.parse_args()
    for path in build(args.processed, args.raw, args.out, args.results_tar):
        print(path)


if __name__ == "__main__":
    main()
