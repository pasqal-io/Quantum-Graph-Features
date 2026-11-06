"""Datasets of the intrinsic benchmark (replaces the original internal loaders).

One graph list per dataset; quantum features are attached node-major
(onebody ``(N, T)``, twobody ``(N, N, T)``) under the attribute names of the
original code (see ``pe_construction.QUANTUM_ATTRS``):

- MUTAG: FM, AFM (emulation, T=20) and HW_AFM (QPU, T=4, ``mutag_hw_afm``);
- ZINC_small: concatenation train/val/test of FM, AFM and the FM/AFM
  reconstruction residuals (11 368 graphs).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from qgf.data import load, load_splits
from qgf.intrinsic.pe_construction import QUANTUM_ATTRS

DATASETS = ("MUTAG", "ZINC_small")

# dataset -> {group: qgf dataset name}
SOURCES: dict[str, dict[str, str]] = {
    "MUTAG": {"fm": "mutag_fm", "afm": "mutag_afm", "hw_afm": "mutag_hw_afm"},
    "ZINC_small": {
        "fm": "zinc_fm",
        "afm": "zinc_afm",
        "fmres": "zinc_fm_residual",
        "afmres": "zinc_afm_residual",
    },
}


def orient(ob: np.ndarray, tb: np.ndarray):
    """Time-major (T, N), (T, N, N) -> node-major (N, T), (N, N, T)."""
    return ob.T, tb.transpose(1, 2, 0)


def _graphs(qgf_name: str, root=None) -> list[Data]:
    kw = {} if root is None else {"root": root}
    if qgf_name.startswith("mutag"):
        return load(qgf_name, **kw)[0]
    splits, _ = load_splits(qgf_name, **kw)
    return splits["train"] + splits["val"] + splits["test"]


def load_dataset(dataset_name: str, root=None) -> list[Data]:
    """Graphs with every quantum feature set of ``dataset_name`` attached (float32 tensors)."""
    if dataset_name not in SOURCES:
        raise ValueError(f"Unknown dataset {dataset_name!r}, expected one of {DATASETS}")
    sources = SOURCES[dataset_name]
    base = _graphs(sources["fm"], root)
    dataset = [
        Data(edge_index=g.edge_index, pos=g.pos, initial_id=g.initial_id, num_nodes=g.num_nodes)
        for g in base
    ]
    for group, qgf_name in sources.items():
        graphs = base if group == "fm" else _graphs(qgf_name, root)
        if len(graphs) != len(dataset):
            raise ValueError(f"{qgf_name}: {len(graphs)} graphs, expected {len(dataset)}")
        ob_attr, tb_attr = QUANTUM_ATTRS[group]
        for d, g in zip(dataset, graphs):
            if int(g.initial_id) != int(d.initial_id) or g.num_nodes != d.num_nodes:
                raise ValueError(f"{qgf_name} misaligned with {sources['fm']}")
            ob, tb = orient(g.onebody_features.numpy(), g.twobody_features.numpy())
            setattr(d, ob_attr, torch.tensor(np.ascontiguousarray(ob), dtype=torch.float32))
            setattr(d, tb_attr, torch.tensor(np.ascontiguousarray(tb), dtype=torch.float32))
    return dataset
