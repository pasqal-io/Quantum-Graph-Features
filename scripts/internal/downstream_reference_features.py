"""INTERNAL ONLY: dump encodings produced by the original graph-transformer code.

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

Runs with the internal venv (DataManager access) and the original repo on sys.path;
writes the transformed onebody/twobody tensors of selected graphs so that the public
port can be checked against them (tests/downstream/test_reference_features.py).

    cd <graph-transformer repo>
    .venv/bin/python <qgf>/scripts/internal/downstream_reference_features.py OUT.pt
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "results/experience_scripts/PE_benchmark_MUTAG")
from graph_transformer.data import (
    PEBenchmarkTransform,
    load_mutag,
    load_mutag_residual,
    load_zinc_small,
    load_zinc_small_residual,
)
from run_pe_benchmark import attach_residuals, make_kfold_indices

MUTAG_CONFIGS = [
    "no_pe",
    "laplacian",
    "rrwp",
    "rrwp_laplacian",
    "af",
    "af_stat",
    "fm",
    "fm_stat",
    "af_fm",
    "af_mean",
    "af_mean_plus_4",
    "fm_mean_plus_5",
    "af_mean_fm_mean",
    "rrwp_af",
    "rrwp_af_stat_fm_stat",
    "rrwp_af_fm",
    "hw_afm",
    "hw_afm_stat",
    "rrwp_hw_afm_stat",
    "rrwp_lap_ring_geo",
    "rrwp_lap_ring_geo_af_fm",
    "rrwp_lap_ring_geo_af_fm_stat",
    "rrwp_lap_ring_geo_fmres_afmres",
    "rrwp_lap_ring_geo_afmres_stat",
]
ZINC_CONFIGS = ["rrwp", "af_stat", "rrwp_af_fm", "rrwp_fm_mean", "rrwp_lap_ring_geo_fmres_afmres"]
FOLD = 3


def pick(n_nodes: list[int]) -> list[int]:
    """First 8 graphs plus every graph with N == 20 (== T, transposition hazard)."""
    return sorted(set(range(8)) | {i for i, n in enumerate(n_nodes) if n == 20})


def dump(splits_t: dict) -> dict:
    out = {}
    for k, lst in splits_t.items():
        idx = pick([d.x.shape[0] for d in lst])[:40]
        out[k] = {
            "idx": idx,
            "ob": [getattr(lst[i], "onebody_features", None) for i in idx],
            "tb": [getattr(lst[i], "twobody_features", None) for i in idx],
        }
    return out


def mutag(cfg: str) -> dict:
    geo = cfg.startswith("rrwp_lap_ring_geo")
    af = cfg in {
        "af",
        "af_stat",
        "af_fm",
        "af_mean",
        "af_mean_plus_4",
        "af_mean_fm_mean",
        "rrwp_af",
        "rrwp_af_stat_fm_stat",
        "rrwp_af_fm",
    }
    fm = cfg in {
        "fm",
        "fm_stat",
        "af_fm",
        "fm_mean_plus_5",
        "af_mean_fm_mean",
        "rrwp_af_stat_fm_stat",
        "rrwp_af_fm",
    }
    hw = cfg.startswith("hw_") or "hw_afm" in cfg
    data_fm = None
    if geo:
        data = load_mutag(pulse="AFM")
        if "fmres" in cfg:
            attach_residuals(
                data, load_mutag_residual("FM"), "onebody_features_fmres", "twobody_features_fmres"
            )
        if "afmres" in cfg:
            attach_residuals(
                data,
                load_mutag_residual("AFM"),
                "onebody_features_afmres",
                "twobody_features_afmres",
            )
        if cfg.endswith(("af_fm", "af_fm_stat")):
            for d in data:
                d.onebody_features_affull = d.onebody_features
                d.twobody_features_affull = d.twobody_features
            attach_residuals(
                data, load_mutag(pulse="FM"), "onebody_features_fmfull", "twobody_features_fmfull"
            )
    elif hw:
        data = load_mutag(pulse="HW_AFM")
    elif af:
        data = load_mutag(pulse="AFM")
        data_fm = load_mutag(pulse="FM") if fm else None
    elif fm:
        data = load_mutag(pulse="FM")
    else:
        data = load_mutag(pulse="AFM")
    folds = make_kfold_indices(data)
    te, va = folds[FOLD], folds[(FOLD + 1) % 10]
    tr = [i for i in range(len(data)) if i not in set(te) | set(va)]
    sp = {k: [data[i] for i in ix] for k, ix in (("train", tr), ("val", va), ("test", te))}
    spf = (
        None
        if data_fm is None
        else {k: [data_fm[i] for i in ix] for k, ix in (("train", tr), ("val", va), ("test", te))}
    )
    topo = not (af or fm or hw) or geo
    t = PEBenchmarkTransform(
        config=cfg,
        af_dataset=sp["train"] if (af or hw) and not geo else None,
        fm_dataset=(spf["train"] if spf else sp["train"]) if fm and not geo else None,
        topo_dataset=sp["train"] if topo else None,
        rrwp_k=20,
        random_dim=20,
    )
    if spf:
        out = {k: [t(d, fm_data=f) for d, f in zip(sp[k], spf[k])] for k in sp}
    else:
        out = {k: [t(d) for d in sp[k]] for k in sp}
    res = dump(out)
    res["fold_test_idx"] = te
    return res


def zinc(cfg: str) -> dict:
    tr, va, te = load_zinc_small(pulse="FM" if cfg == "rrwp_fm_mean" else "AFM")
    sp = {"train": tr, "val": va, "test": te}
    if cfg.startswith("rrwp_lap_ring_geo"):
        for name, attr in (("FM", "fmres"), ("AFM", "afmres")):
            r = dict(zip(("train", "val", "test"), load_zinc_small_residual(name)))
            for k in sp:
                attach_residuals(
                    sp[k], r[k], f"onebody_features_{attr}", f"twobody_features_{attr}"
                )
        t = PEBenchmarkTransform(config=cfg, topo_dataset=sp["train"], rrwp_k=20, random_dim=20)
        return dump({k: [t(d) for d in sp[k]] for k in sp})
    if cfg == "rrwp_af_fm":
        ftr, fva, fte = load_zinc_small(pulse="FM")
        spf = {"train": ftr, "val": fva, "test": fte}
        t = PEBenchmarkTransform(config=cfg, af_dataset=tr, fm_dataset=ftr, rrwp_k=20)
        return dump({k: [t(d, f) for d, f in zip(sp[k], spf[k])] for k in sp})
    kw = {
        "rrwp": {"topo_dataset": tr},
        "af_stat": {"af_dataset": tr},
        "rrwp_fm_mean": {"fm_dataset": tr},
    }[cfg]
    t = PEBenchmarkTransform(config=cfg, rrwp_k=20, **kw)
    return dump({k: [t(d) for d in sp[k]] for k in sp})


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    ref = {"mutag": {c: mutag(c) for c in MUTAG_CONFIGS}}
    ref["zinc"] = {c: zinc(c) for c in ZINC_CONFIGS}
    torch.save(ref, sys.argv[1])
    print("saved", sys.argv[1])
