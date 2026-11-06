"""Dataset loading for the downstream GT benchmark (MUTAG and ZINC*_subset).

Replaces the dataset loaders of the original internal implementation by
:func:`qgf.data.load`. Quantum features of each needed source are
attached to the base graphs as ``onebody_<src>`` [T, N] and ``twobody_<src>`` [T, N, N].

Scale: the released files store correlations in the ZZ convention (4 C). The original
pipeline consumed some sources without the factor 4 (ZINC AFM/FM/residuals and MUTAG
HW-AFM). We divide by ``meta["zz_scale_applied"]`` so the tensors are exactly those of
the original run; since every block is z-scored this only matters through the 1e-6 std
epsilon, but it keeps the port bit-faithful. With that undone, the original MUTAG
``clamp(-1, 1)`` is a no-op (max |value| = 1.0 for AFM/FM, 0.33 for HW-AFM) and is
therefore not reproduced, except for ``hw_afm_3t`` (ESA hardware rows of Table 4): those
runs read the hardware values in the ZZ convention, where the clip is active (one-body
values up to 1.3), so this source keeps the factor 4 and is clipped to [-1, 1].
"""

from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Batch, Data

from qgf.data import DEFAULT_ROOT, load, load_splits

MUTAG_NUM_ATOM_TYPES = 7
MUTAG_NUM_BOND_TYPES = 4
ZINC_NUM_ATOM_TYPES = 28
ZINC_NUM_BOND_TYPES = 4

# quantum source -> released dataset name prefix
_SOURCE_DATASET = {
    "afm": "afm",
    "fm": "fm",
    "hw_afm": "hw_afm",
    "afm_res": "afm_residual",
    "fm_res": "fm_residual",
    "hw_afm_3t": "hw_afm",
    "afm_exact": "exact_afm",
    "fm_exact": "exact_fm",
    "afm_exact_nn": "exact_nn_afm",
    "fm_exact_nn": "exact_nn_fm",
}
# Sources restricted to some measurement times (ns): the ESA hardware runs of Table 4 used
# the hardware set without the 715 ns run (identical values at the three other times).
_SOURCE_TIMES = {"hw_afm_3t": (305, 512, 4000)}
# Sources consumed as stored (ZZ convention, 4 C) and clipped to [-1, 1], as the loader of
# the ESA hardware runs did with its copy of these values: the clip is active there (many
# one-body values exceed 1) and changes the z-scored features.
_SOURCE_ZZ_CLIPPED = {"hw_afm_3t"}
# MUTAG subsets: "exact173" = the graphs with N <= 24 (exact state-vector features exist only
# for them); folds are rebuilt on the subset, as in the original runs.
MUTAG_SUBSETS = {"exact173": 24}


def _attach(base: list[Data], other: list[Data], meta: dict, source: str) -> None:
    """Attach ``other``'s quantum features to ``base`` (aligned by index + initial_id)."""
    if len(base) != len(other):
        raise ValueError(f"{source}: {len(other)} graphs, base has {len(base)}")
    clip = source in _SOURCE_ZZ_CLIPPED
    scale = 1.0 if clip else float(meta["zz_scale_applied"])
    times = _SOURCE_TIMES.get(source)
    idx = None if times is None else [list(meta["times_ns"]).index(t) for t in times]
    for b, o in zip(base, other):
        if int(b.initial_id) != int(o.initial_id):
            raise ValueError(f"{source}: misaligned graphs (initial_id mismatch)")
        ob, tb = o.onebody_features, o.twobody_features
        if idx is not None:
            ob, tb = ob[idx], tb[idx]
        ob, tb = ob / scale, tb / scale
        if clip:
            ob, tb = ob.clamp(-1.0, 1.0), tb.clamp(-1.0, 1.0)
        setattr(b, f"onebody_{source}", ob)  # [T, N]
        setattr(b, f"twobody_{source}", tb)  # [T, N, N]


def _strip(graphs: list[Data]) -> list[Data]:
    """Keep topology, labels and positions; quantum features come via ``_attach``."""
    return [
        Data(
            x=g.x,
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            y=g.y,
            pos=g.pos,
            initial_id=g.initial_id,
        )
        for g in graphs
    ]


def load_mutag(
    sources: list[str], root: Path = DEFAULT_ROOT, subset: str | None = None
) -> list[Data]:
    """188 MUTAG graphs (dataset order) with the requested quantum sources attached.

    As in the original loader, one-hot atom / bond features become indices and ``y``
    is an int64 class label. ``subset`` (:data:`MUTAG_SUBSETS`) keeps the small graphs.
    """
    max_nodes = None if subset is None else MUTAG_SUBSETS[subset]

    def keep(gs: list[Data]) -> list[Data]:  # full-size sources are restricted as the base
        return gs if max_nodes is None else [g for g in gs if g.num_nodes <= max_nodes]

    raw, _ = load("mutag_afm", root=root)
    graphs = _strip(keep(raw))
    for g in graphs:
        g.x = g.x.argmax(dim=-1)  # [N, 7] -> [N]
        g.edge_attr = g.edge_attr.argmax(dim=-1)  # [E, 4] -> [E]
        g.y = g.y.long()
    for src in sources:
        other, meta = load(f"mutag_{_SOURCE_DATASET[src]}", root=root)
        _attach(graphs, keep(other), meta, src)
    return graphs


def load_zinc(sources: list[str], root: Path = DEFAULT_ROOT) -> dict[str, list[Data]]:
    """ZINC*_subset train/val/test (9475/946/947) with the requested sources attached."""
    raw, _ = load_splits("zinc_afm", root=root)
    splits = {k: _strip(v) for k, v in raw.items()}
    for graphs in splits.values():
        for g in graphs:
            g.y = g.y.float()  # original: float32 target
    for src in sources:
        other, meta = load_splits(f"zinc_{_SOURCE_DATASET[src]}", root=root)
        for k, graphs in splits.items():
            _attach(graphs, other[k], meta, src)
    return splits


def custom_collate_with_pe(data_list: list[Data]) -> Batch:
    """Batch graphs while keeping per-graph PE tensors as lists (variable shapes)."""
    onebody_list: list = []
    twobody_list: list = []
    for data in data_list:
        ob = getattr(data, "onebody_features", None)
        tb = getattr(data, "twobody_features", None)
        if ob is not None:
            onebody_list.append(ob)
            delattr(data, "onebody_features")
        if tb is not None:
            twobody_list.append(tb)
            delattr(data, "twobody_features")

    batch = Batch.from_data_list(data_list)
    if onebody_list:
        batch.onebody_features = onebody_list
    if twobody_list:
        batch.twobody_features = twobody_list

    # Restore on original data objects (they are reused every epoch)
    for i, data in enumerate(data_list):
        if i < len(onebody_list):
            data.onebody_features = onebody_list[i]
        if i < len(twobody_list):
            data.twobody_features = twobody_list[i]
    return batch


__all__ = [
    "MUTAG_NUM_ATOM_TYPES",
    "MUTAG_NUM_BOND_TYPES",
    "ZINC_NUM_ATOM_TYPES",
    "ZINC_NUM_BOND_TYPES",
    "custom_collate_with_pe",
    "load_mutag",
    "load_zinc",
]
