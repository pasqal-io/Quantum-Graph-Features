"""Plain on-disk format for QGF datasets.

A dataset file is ``torch.save`` of a dict holding only plain containers and
tensors, so it loads with ``torch.load(weights_only=True)`` and needs no
project-specific class.
"""

from pathlib import Path

import torch
from torch_geometric.data import Data

FORMAT_VERSION = 1


def save_dataset(path: str | Path, graphs: list[Data], meta: dict) -> None:
    plain = []
    for i, g in enumerate(graphs):
        attrs = {}
        for key, value in g.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"graph {i}: attribute {key!r} is {type(value).__name__}, not a tensor"
                )
            attrs[key] = value
        plain.append(attrs)
    torch.save({"format_version": FORMAT_VERSION, "meta": meta, "graphs": plain}, path)


def load_dataset(path: str | Path) -> tuple[list[Data], dict]:
    blob = torch.load(path, weights_only=True)
    if blob["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {blob['format_version']}, expected {FORMAT_VERSION}"
        )
    return [Data(**attrs) for attrs in blob["graphs"]], blob["meta"]
