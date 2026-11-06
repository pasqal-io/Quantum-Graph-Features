"""Downstream GNN benchmark: run trials (Tables 2-3, GCN / GAT / GINE / Transformer / General).

Each trial writes ``<out>/<dataset>/<model>/<config>/fold_<k>/seed_<s>/results.json``
(MUTAG) or ``<out>/<dataset>/<model>/<config>/seed_<s>/results.json`` (ZINC). Finished
trials are skipped, so the command can be re-run and split across jobs with ``--chunk i/n``.

    # one MUTAG trial
    uv run python experiments/downstream_gnns/run.py --dataset mutag --models GCN \
        --configs rrwp --folds 0 --seeds 1
    # the whole MUTAG grid, chunk 3 of 100 (see submit.sh)
    uv run python experiments/downstream_gnns/run.py --dataset mutag --chunk 3/100
"""

from __future__ import annotations

import argparse
import itertools
import os
import traceback
from pathlib import Path

from qgf.downstream.gnns.benchmark import MUTAG_SEEDS, ZINC_SEEDS, run_trial
from qgf.downstream.gnns.model import MODELS
from qgf.downstream.gt.benchmark import N_FOLDS
from qgf.downstream.gt.pe_transforms import CONFIGS

TABLE_CONFIGS = [  # Tables 2 and 3
    "no_pe",
    "random",
    "laplacian",
    "rrwp",
    "rrwp_laplacian",
    "af",
    "af_stat",
    "fm",
    "fm_stat",
    "af_fm",
    "rrwp_af",
    "rrwp_fm",
    "rrwp_af_stat",
    "rrwp_fm_stat",
    "rrwp_af_fm",
    "rrwp_af_stat_fm_stat",
]
HW_CONFIGS = ["hw_afm", "hw_afm_stat", "rrwp_hw_afm", "rrwp_hw_afm_stat"]  # MUTAG only
GEO_CONFIGS = ["rrwp_lap_ring_geom", "rrwp_lap_ring_geom_af_fm"]  # geometry-baseline table
GRIDS = {
    "mutag": TABLE_CONFIGS + HW_CONFIGS + GEO_CONFIGS,
    "zinc": TABLE_CONFIGS + GEO_CONFIGS,
}
# Cells re-run to validate the port (all folds x seeds of each): (backbone, config).
VALIDATION = {
    "mutag": [
        ("GCN", "rrwp"),
        ("GCN", "rrwp_af"),
        ("GINE", "rrwp_af_fm"),
        ("GAT", "rrwp_fm_stat"),
        ("Transformer", "rrwp_lap_ring_geom"),
        ("Transformer", "rrwp_lap_ring_geom_af_fm"),
        ("General", "rrwp_hw_afm_stat"),
    ],
    "zinc": [("GINE", "rrwp"), ("GAT", "rrwp_af_stat_fm_stat")],
}
# Fixed champions of the App. G statistics (classical RRWP vs hybrid RRWP+AFMstat+FMstat).
CHAMPIONS = [(m, c) for m in MODELS for c in ("rrwp", "rrwp_af_stat_fm_stat")]


def trial_dir(
    out: Path, dataset: str, model: str, config: str, fold: int | None, seed: int
) -> Path:
    d = out / dataset / model / config
    return (d / f"fold_{fold}" if fold is not None else d) / f"seed_{seed}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mutag", "zinc"], required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["paper"],
        help="config names, 'paper' (full grid), 'validation' or 'champions' (preset cells)",
    )
    p.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("results/downstream_gnns"))
    p.add_argument("--chunk", default="0/1", help="i/n: run every n-th trial starting at i")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args(argv)

    seeds = args.seeds or (MUTAG_SEEDS if args.dataset == "mutag" else ZINC_SEEDS)
    folds = args.folds if args.dataset == "mutag" else [None]
    if args.configs == ["validation"]:
        cells = VALIDATION[args.dataset]
    elif args.configs == ["champions"]:
        cells = CHAMPIONS
    else:
        configs = GRIDS[args.dataset] if args.configs == ["paper"] else args.configs
        cells = list(itertools.product(args.models, configs))
    unknown = {c for _, c in cells} - set(CONFIGS)
    if unknown:
        p.error(f"unknown configs: {sorted(unknown)}")
    trials = [(m, c, f, s) for m, c in cells for f in folds for s in seeds]
    i, n = (int(v) for v in os.path.expandvars(args.chunk).split("/"))
    kw = {} if args.data_root is None else {"root": args.data_root}

    failed = 0
    metric = "test_f1_macro" if args.dataset == "mutag" else "test_mae"
    for model, config, fold, seed in trials[i::n]:
        out_dir = trial_dir(args.out, args.dataset, model, config, fold, seed)
        if (out_dir / "results.json").exists():
            continue
        print(f"=== {args.dataset} {model} {config} fold={fold} seed={seed}", flush=True)
        try:
            r = run_trial(
                args.dataset,
                model,
                config,
                seed,
                fold,
                out_dir,
                max_epochs=args.max_epochs,
                **kw,
            )
            print(f"=== done {model} {config} fold={fold} seed={seed} {metric}={r[metric]:.4f}")
        except Exception:  # keep going with the other trials of the chunk
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
