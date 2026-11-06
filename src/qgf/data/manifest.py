"""sha256 manifest for the released data files."""

import hashlib
import json
from pathlib import Path

MANIFEST = "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path, entries: dict[str, int]) -> Path:
    out = root / MANIFEST
    if out.exists():  # merge: partial re-exports keep the other entries
        old = json.loads(out.read_text())
        entries = {n: e["n_graphs"] for n, e in old.items() if (root / n).exists()} | entries
    manifest = {name: {"sha256": sha256(root / name), "n_graphs": n} for name, n in entries.items()}
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def verify_manifest(root: Path) -> list[str]:
    manifest = json.loads((root / MANIFEST).read_text())
    problems = []
    for name, entry in manifest.items():
        path = root / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"{name}: checksum mismatch")
    return problems
