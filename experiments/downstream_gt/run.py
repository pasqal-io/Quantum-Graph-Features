"""Downstream Graph Transformer benchmark: run trials (paper Tables 2-3, GT column).

Each trial writes ``<out>/<dataset>/<config>/fold_<k>/seed_<s>/results.json`` (MUTAG)
or ``<out>/<dataset>/<config>/seed_<s>/results.json`` (ZINC). Finished trials are
skipped, so the command can be re-run / split across jobs with ``--chunk i/n``.

Examples:
    # one MUTAG trial
    uv run python experiments/downstream_gt/run.py --dataset mutag --configs rrwp \
        --folds 0 --seeds 1
    # the whole MUTAG paper grid, chunk 3 of 40 (see submit.sh)
    uv run python experiments/downstream_gt/run.py --dataset mutag --configs paper \
        --chunk 3/40
"""

from __future__ import annotations

import argparse
import itertools
import traceback
from pathlib import Path

from qgf.downstream.gt.benchmark import MUTAG_SEEDS, N_FOLDS, ZINC_SEEDS, run_trial
from qgf.downstream.gt.pe_transforms import CONFIGS

# Feature sets of the paper, GT column.
TABLE_CONFIGS = [  # Tables 2 and 3
    "no_pe",
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
EXTRA_CONFIGS = [  # Random baseline and time-mean variants (appendix / revised tables)
    "random",
    "af_mean",
    "fm_mean",
    "af_mean_fm_mean",
    "rrwp_af_mean",
    "rrwp_fm_mean",
    "rrwp_af_mean_fm_mean",
]
HW_CONFIGS = ["hw_afm", "hw_afm_stat", "rrwp_hw_afm", "rrwp_hw_afm_stat"]  # MUTAG only
GEO_CONFIGS = ["rrwp_lap_ring_geo", "rrwp_lap_ring_geo_af_fm"]  # geometry-baseline table
GRIDS = {
    "mutag": TABLE_CONFIGS + EXTRA_CONFIGS + HW_CONFIGS + GEO_CONFIGS,
    "zinc": TABLE_CONFIGS + EXTRA_CONFIGS,
}


def trial_dir(out: Path, dataset: str, config: str, fold: int | None, seed: int) -> Path:
    d = out / dataset / config
    return (d / f"fold_{fold}" if fold is not None else d) / f"seed_{seed}"


def finished(out_dir: Path) -> bool:
    """True if the trial already has a ``results.json``."""
    return (out_dir / "results.json").exists()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["mutag", "zinc"], required=True)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["paper"],
        help="config names, or 'paper' for the full paper grid of the dataset",
    )
    p.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("results/downstream_gt"))
    p.add_argument("--chunk", default="0/1", help="i/n: run every n-th trial starting at i")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args()

    configs = GRIDS[args.dataset] if args.configs == ["paper"] else args.configs
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        p.error(f"unknown configs: {sorted(unknown)}")
    seeds = args.seeds or (MUTAG_SEEDS if args.dataset == "mutag" else ZINC_SEEDS)
    folds = args.folds if args.dataset == "mutag" else [None]
    trials = list(itertools.product(configs, folds, seeds))
    i, n = (int(v) for v in args.chunk.split("/"))
    kw = {} if args.data_root is None else {"root": args.data_root}

    failed = 0
    for config, fold, seed in trials[i::n]:
        out_dir = trial_dir(args.out, args.dataset, config, fold, seed)
        if finished(out_dir):
            continue
        print(f"=== {args.dataset} {config} fold={fold} seed={seed}", flush=True)
        try:
            r = run_trial(
                args.dataset,
                config,
                seed,
                fold,
                out_dir,
                max_epochs=args.max_epochs,
                **kw,
            )
            metric = "test_f1_macro" if args.dataset == "mutag" else "test_mae"
            print(f"=== done {config} fold={fold} seed={seed} {metric}={r[metric]:.4f}", flush=True)
        except Exception:  # keep going with the other trials of the chunk
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
