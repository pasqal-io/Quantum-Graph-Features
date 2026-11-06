"""Export internal QGF datasets to the public plain format.

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

It needs Pasqal's ``data_manager`` and the ``quantum_leap``
classes to unpickle the sources. Run it from the graph-transformer venv on the
cluster. Everything it writes is readable with public packages only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from qgf.data.io import save_dataset
from qgf.data.manifest import write_manifest
from qgf.data.registry import file_name

EMULATION_TIMES_NS = [int(t) for t in np.linspace(100, 4000, 20, dtype=int)]
# Nominal durations of the four QPU runs (305, 512, 715 ns padded, 4 us).
QPU_TIMES_NS = [305, 512, 715, 4000]
RENAMES = {"initial_ds_id": "initial_id"}
# ZINC_tiny (ablations, Tables 8-10): earlier generation protocol, see ZINC_TINY_NOTE.
ZINC_TINY_URI = "ZINC/unbiased_1500_subset_on_pattern_with_quantum_features/{pulse}/{{split}}"
ZINC_TINY_NOTE = (
    "Ablation dataset of Tables 8-10 (ZINC_tiny, 1.5k graphs), generated with an earlier "
    "protocol than the other datasets: one constant global pulse of 6000 ns with the same "
    "(Omega, delta) for every graph, (18, 0) rad/us for the AFM-like and (27, 62) rad/us for "
    "the FM-like set, 1000-shot estimates at 20 measurement times inside the pulse (the time "
    "grid was not stored). The source stores the occupations <n_i> as one-body features and "
    "the connected correlations C as two-body features; here onebody_features = 4 C_ii, "
    "twobody_features = 4 C (zero diagonal) and `occupation` = <n_i> [T, N]."
)
# datasets whose source one-body block holds occupations (see convert_graph)
OCCUPATION_SOURCES = frozenset({"zinc_tiny_afm", "zinc_tiny_fm"})
QUANTUM_KEYS = ("onebody_features", "twobody_features")

# name -> (project, uri template, version, zz_scale, onebody_is_nt, times or None)
SOURCES: dict[str, tuple[str, str, int, float, bool, list[int] | None]] = {
    "mutag_afm": ("results", "MUTAG/AFM/qcorr_dataset", 0, 1.0, False, EMULATION_TIMES_NS),
    "mutag_fm": ("results", "MUTAG/FM/qcorr_dataset", 0, 1.0, False, EMULATION_TIMES_NS),
    "mutag_hw_afm": ("results", "MUTAG/QPU/qcorr_dataset", 4, 4.0, False, QPU_TIMES_NS),
    "mutag_afm_residual": (
        "results",
        "MUTAG/AFM_residual/qcorr_dataset",
        0,
        1.0,
        True,
        EMULATION_TIMES_NS,
    ),
    "mutag_fm_residual": (
        "results",
        "MUTAG/FM_residual/qcorr_dataset",
        0,
        1.0,
        True,
        EMULATION_TIMES_NS,
    ),
    "zinc_afm": (
        "results",
        "ZINC_small/{split}/AFM/qcorr_dataset",
        2,
        4.0,
        False,
        EMULATION_TIMES_NS,
    ),
    "zinc_fm": (
        "results",
        "ZINC_small/{split}/FM/qcorr_dataset",
        0,
        4.0,
        False,
        EMULATION_TIMES_NS,
    ),
    "zinc_afm_residual": (
        "results",
        "ZINC_small/{split}/AFM_residual/qcorr_dataset",
        0,
        4.0,
        True,
        EMULATION_TIMES_NS,
    ),
    "zinc_fm_residual": (
        "results",
        "ZINC_small/{split}/FM_residual/qcorr_dataset",
        0,
        4.0,
        True,
        EMULATION_TIMES_NS,
    ),
    "zinc1500_afm": (
        "qgml_pioneer",
        "ZINC1500/with_features/AFM/{split}",
        2,
        4.0,
        False,
        EMULATION_TIMES_NS,
    ),
    "zinc1500_fm": (
        "qgml_pioneer",
        "ZINC1500/with_features/FM/{split}",
        2,
        4.0,
        False,
        EMULATION_TIMES_NS,
    ),
    "zinc_tiny_afm": (
        "qet_small_subsets",
        ZINC_TINY_URI.format(pulse="6000_18.000_0.000"),
        0,
        4.0,
        False,
        None,
    ),
    "zinc_tiny_fm": (
        "qet_small_subsets",
        ZINC_TINY_URI.format(pulse="6000_27.000_62.000"),
        0,
        4.0,
        False,
        None,
    ),
}


FAMILIES = {
    "mutag": ("mutag_afm", "mutag_fm", "mutag_hw_afm", "mutag_afm_residual", "mutag_fm_residual"),
    "zinc": ("zinc_afm", "zinc_fm", "zinc_afm_residual", "zinc_fm_residual"),
    "zinc1500": ("zinc1500_afm", "zinc1500_fm"),
    "zinc_tiny": ("zinc_tiny_afm", "zinc_tiny_fm"),
}


def kind(name: str) -> str:
    if name == "mutag_hw_afm":
        return "qpu"
    return "residual" if name.endswith("_residual") else "emulation"


def make_meta(name: str, split: str | None, scale: float, times: list[int] | None) -> dict:
    meta = {
        "name": name,
        "split": split,
        "times_ns": times,
        "zz_scale_applied": scale,
        "kind": kind(name),
    }
    if meta["kind"] == "qpu":
        meta["time_note"] = (
            "Nominal durations of the four QPU runs (labels T2, T2.5, T4, T15). "
            "Only 305 and 715 ns are emulation-equivalent times."
        )
    if name in OCCUPATION_SOURCES:
        meta["time_note"] = ZINC_TINY_NOTE
    return meta


def check_alignment(ids: dict[str, list[int]]) -> None:
    """Raise if the datasets do not all share the same ``initial_id`` sequence."""
    ref_name, ref = next(iter(ids.items()))
    bad = [n for n, v in ids.items() if v != ref]
    if bad:
        raise ValueError(
            f"initial_id mismatch vs {ref_name}: {bad} ({ {n: len(ids[n]) for n in ids} })"
        )


def _to_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.dtype.kind not in "biuf":
        return None
    return torch.from_numpy(np.ascontiguousarray(arr))


def convert_graph(g, *, zz_scale: float, onebody_is_nt: bool, occupation: bool = False) -> Data:
    attrs = {}
    for key, value in g.items():
        tensor = _to_tensor(value)
        if tensor is not None:
            attrs[RENAMES.get(key, key)] = tensor
    if onebody_is_nt:
        attrs["onebody_features"] = attrs["onebody_features"].t().contiguous()
    attrs.pop("threebody_features", None)
    if occupation:  # source one-body = <n_i> [T, N], two-body = C [T, N, N] with C_ii = n(1-n)
        attrs["occupation"] = attrs["onebody_features"].float()
        corr = attrs["twobody_features"].float()
        attrs["onebody_features"] = torch.diagonal(corr, dim1=1, dim2=2).clone()  # [T, N]
        idx = torch.arange(corr.shape[1])
        corr[:, idx, idx] = 0.0
        attrs["twobody_features"] = corr
    for key in QUANTUM_KEYS:
        attrs[key] = attrs[key].float() * zz_scale
    if "embedded_position" in attrs:
        emb = attrs.pop("embedded_position").float()
        if "pos" in attrs:
            assert torch.allclose(attrs["pos"].float(), emb, atol=1e-4), "pos != embedded_position"
        attrs["pos"] = emb
    elif "pos" in attrs:
        attrs["pos"] = attrs["pos"].float()
    return Data(**attrs)


def export(name: str, out: Path, dm_cls) -> tuple[dict[str, int], dict[str, list[int]]]:
    project, uri, version, scale, onebody_is_nt, times = SOURCES[name]
    splits = ("train", "val", "test") if "{split}" in uri else (None,)
    counts, ids = {}, {}
    for split in splits:
        source = uri.format(split=split)
        occupation = name in OCCUPATION_SOURCES
        graphs = [
            convert_graph(g, zz_scale=scale, onebody_is_nt=onebody_is_nt, occupation=occupation)
            for g in dm_cls(project).load(source, version=version)
        ]
        if times is not None:
            assert all(g.twobody_features.shape[0] == len(times) for g in graphs), name
        meta = make_meta(name, split, scale, times)
        fname = file_name(name, split)
        save_dataset(out / fname, graphs, meta)
        counts[fname] = len(graphs)
        ids[fname] = [int(g.initial_id.flatten()[0]) for g in graphs]
        print(f"{fname}: {len(graphs)} graphs")
    return counts, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = parser.parse_args()

    from data_manager import DataManager  # internal package

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    ids: dict[str, list[int]] = {}
    for name in args.only:
        c, i = export(name, args.out, DataManager)
        counts.update(c)
        ids.update(i)
    for family, members in FAMILIES.items():
        for split in (None, "train", "val", "test"):
            suffix = "" if split is None else f"_{split}"
            group = {n: ids[f"{n}{suffix}.pt"] for n in members if f"{n}{suffix}.pt" in ids}
            if len(group) > 1:
                check_alignment(group)
                print(
                    f"aligned {family}/{split}: {sorted(group)} "
                    f"({len(next(iter(group.values())))} graphs)"
                )
    write_manifest(args.out, counts)


if __name__ == "__main__":
    main()
