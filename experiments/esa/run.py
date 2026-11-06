"""ESA benchmark on MUTAG: run trials (paper Table 4, concatenation vs gated fusion).

Each trial writes ``<out>/<fusion>/<config>/fold_<k>/seed_<s>/results.json``; finished
trials are skipped, so the command can be re-run and split across jobs (``--chunk i/n``).

    # one trial (GPU strongly recommended: ~5 min on an A100)
    uv run --extra esa python experiments/esa/run.py --configs af_fm --fusions gated \
        --folds 0 --seeds 1
    # the Table 4 grid (15 cells x 10 folds x 5 seeds)
    uv run --extra esa python experiments/esa/run.py --configs paper
"""

from __future__ import annotations

import argparse
import itertools
import os
import traceback
from pathlib import Path

from qgf.esa.benchmark import CONFIGS, FUSIONS, SEEDS, run_trial

# (fusion, config) cells of Table 4; the baseline (RWSE*, no quantum) has no gate.
PAPER = [("concat", "no_pe")] + [
    (f, c)
    for c in ("af_stat", "fm_stat", "af", "fm", "af_fm", "hw_afm_3t", "hw_afm_3t_stat")
    for f in FUSIONS
]
# Cells re-run to validate the port.
VALIDATION = [("concat", "no_pe"), ("concat", "af_fm"), ("gated", "af_fm"), ("gated", "af_stat")]


def trial_dir(out: Path, fusion: str, config: str, fold: int, seed: int) -> Path:
    return out / fusion / config / f"fold_{fold}" / f"seed_{seed}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--configs",
        nargs="+",
        default=["paper"],
        help="configs, 'paper' (Table 4 cells) or 'validation'",
    )
    p.add_argument("--fusions", nargs="+", default=FUSIONS, choices=FUSIONS)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--out", type=Path, default=Path("results/esa"))
    p.add_argument("--chunk", default="0/1", help="i/n: run every n-th trial starting at i")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument(
        "--attention",
        choices=["sdpa", "xformers"],
        default="sdpa",
        help="xformers = kernel of the paper runs (use with reproduce/paper_env)",
    )
    args = p.parse_args(argv)

    if args.configs == ["paper"]:
        cells = PAPER
    elif args.configs == ["validation"]:
        cells = VALIDATION
    else:
        unknown = set(args.configs) - set(CONFIGS)
        if unknown:
            p.error(f"unknown configs: {sorted(unknown)}")
        cells = [(f, c) for c in args.configs for f in args.fusions]
    trials = [(f, c, k, s) for (f, c), k, s in itertools.product(cells, args.folds, args.seeds)]
    i, n = (int(v) for v in os.path.expandvars(args.chunk).split("/"))
    kw = {"attention": args.attention}
    if args.data_root is not None:
        kw["root"] = args.data_root
    failed = 0
    for fusion, config, fold, seed in trials[i::n]:
        out_dir = trial_dir(args.out, fusion, config, fold, seed)
        if (out_dir / "results.json").exists():
            continue
        print(f"=== ESA {fusion} {config} fold={fold} seed={seed}", flush=True)
        try:
            r = run_trial(config, fusion, seed, fold, out_dir, max_epochs=args.max_epochs, **kw)
            print(f"=== done {fusion} {config} fold={fold} seed={seed} f1={r['test_f1']:.4f}")
        except Exception:  # keep going with the other trials of the chunk
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
