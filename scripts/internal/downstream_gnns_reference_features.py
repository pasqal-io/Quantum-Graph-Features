"""INTERNAL ONLY: dump the encodings fed to the original GNN runs.

Internal use only: requires the internal repository and DataManager access; not needed to
reproduce results from the released data.

Calls the original ``load_and_transform_mutag`` / ``load_and_transform_zinc_small`` of the
PE dataloader template (the functions the GNN runs used) and stores the transformed
onebody/twobody tensors of selected graphs, keyed by the port's config names, so that the
port can be checked against them (tests/downstream/test_gnns_reference_features.py).

    cd <qet repository>
    .venv/bin/python <qgf>/scripts/internal/downstream_gnns_reference_features.py \
        --template qet/graph_transformer/PE_dataloader_template --out OUT.pt
"""

import argparse
import sys
from pathlib import Path

import torch

FOLD = 3
# original config name -> port config name
MUTAG = {
    "no_pe": "no_pe",
    "laplacian": "laplacian",
    "rrwp": "rrwp",
    "rrwp_laplacian": "rrwp_laplacian",
    "af": "af",
    "af_stat": "af_stat",
    "fm": "fm",
    "fm_stat": "fm_stat",
    "af_fm": "af_fm",
    "rrwp_af": "rrwp_af",
    "rrwp_fm_stat": "rrwp_fm_stat",
    "rrwp_af_fm": "rrwp_af_fm",
    "rrwp_af_stat_fm_stat": "rrwp_af_stat_fm_stat",
    "afm_qpu": "hw_afm",
    "rrwp_afm_qpu_stat": "rrwp_hw_afm_stat",
    "rrwp_laplacian_ring_geom": "rrwp_lap_ring_geom",
    "rrwp_laplacian_ring_geom_af_fm": "rrwp_lap_ring_geom_af_fm",
}
ZINC = {
    "rrwp": "rrwp",
    "af_stat": "af_stat",
    "rrwp_af_fm": "rrwp_af_fm",
    "rrwp_laplacian_ring_geom_af_fm": "rrwp_lap_ring_geom_af_fm",
}


def pick(n_nodes: list[int]) -> list[int]:
    """First 8 graphs plus every graph with N == 20 (== T, transposition hazard)."""
    return sorted(set(range(8)) | {i for i, n in enumerate(n_nodes) if n == 20})[:40]


def dump(splits: dict) -> dict:
    out = {}
    for k, lst in splits.items():
        idx = pick([d.x.shape[0] for d in lst])
        out[k] = {
            "idx": idx,
            "initial_id": [int(lst[i].initial_id.flatten()[0]) for i in idx],
            "ob": [getattr(lst[i], "onebody_features", None) for i in idx],
            "tb": [getattr(lst[i], "twobody_features", None) for i in idx],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--template", type=Path, required=True, help="PE_dataloader_template dir")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(args.template.resolve()))
    from pe_data_utils import load_and_transform_mutag, load_and_transform_zinc_small

    ref: dict = {"mutag": {}, "zinc": {}, "fold": FOLD}
    for orig, name in MUTAG.items():
        tr, va, te, *_ = load_and_transform_mutag(config=orig, fold_id=FOLD)
        ref["mutag"][name] = dump({"train": tr, "val": va, "test": te})
    for orig, name in ZINC.items():
        tr, va, te, *_ = load_and_transform_zinc_small(config=orig)
        ref["zinc"][name] = dump({"train": tr, "val": va, "test": te})
    torch.save(ref, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
