"""Names of the released datasets and helpers to load them."""

import os
from pathlib import Path

from torch_geometric.data import Data

from qgf.data.io import load_dataset
from qgf.data.manifest import verify_manifest

DEFAULT_ROOT = Path(os.environ.get("QGF_DATA", "data/processed"))
SPLITS = ("train", "val", "test")
SINGLE_DATASETS = (
    "mutag_afm",
    "mutag_fm",
    "mutag_hw_afm",
    "mutag_afm_residual",
    "mutag_fm_residual",
    "ladder_afm",
    "ladder_fm",
    "mutag_exact_afm",
    "mutag_exact_fm",
    "mutag_exact_nn_afm",
    "mutag_exact_nn_fm",
)
SPLIT_DATASETS = (
    "zinc_afm",
    "zinc_fm",
    "zinc_afm_residual",
    "zinc_fm_residual",
    "zinc1500_afm",
    "zinc1500_fm",
    "zinc_tiny_afm",
    "zinc_tiny_fm",
)


def file_name(name: str, split: str | None = None) -> str:
    if name in SINGLE_DATASETS:
        if split is not None:
            raise ValueError(f"{name} has no splits")
        return f"{name}.pt"
    if name in SPLIT_DATASETS:
        if split not in SPLITS:
            raise ValueError(f"{name} needs split in {SPLITS}, got {split!r}")
        return f"{name}_{split}.pt"
    raise ValueError(f"unknown dataset {name!r}")


def load(name: str, split: str | None = None, root: Path = DEFAULT_ROOT) -> tuple[list[Data], dict]:
    return load_dataset(Path(root) / file_name(name, split))


def load_splits(name: str, root: Path = DEFAULT_ROOT) -> tuple[dict[str, list[Data]], dict]:
    """Return ``({split: graphs}, meta)``; meta is the train split's (all splits share times_ns)."""
    loaded = {split: load(name, split, root) for split in SPLITS}
    return {s: g for s, (g, _) in loaded.items()}, loaded["train"][1]


def verify(root: Path = DEFAULT_ROOT) -> list[str]:
    return verify_manifest(Path(root))
