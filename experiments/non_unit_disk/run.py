"""Exact non-unit-disk benchmark on MUTAG: run GNN trials on the 173-graph subset.

Four configs share the strong classical block (RRWP + Laplacian + ring + register geometry,
35 / 33 dims) and differ in the quantum block (40 / 40 dims): none, the released AFM + FM
emulation (1000 shots), exact AFM + FM of the register couplings ``C6 / r^6`` (``_exact``),
exact AFM + FM of the graph Hamiltonian ``J_NN A`` (``_exact_nn``). Seven backbones, 10
stratified folds rebuilt on the subset (``subset="exact173"``), seeds 1-5, training as
Table 2. Each trial writes ``<out>/<model>/<config>/fold_<k>/seed_<s>/results.json``.

    uv run python experiments/non_unit_disk/run.py --models GCN --configs \\
        rrwp_lap_ring_geom_exact_nn --folds 0 --seeds 1
    NTASKS=4 sbatch --array=0-3 experiments/non_unit_disk/submit.sh       # 1400 trials
"""

from __future__ import annotations

import argparse
import itertools
import traceback
from pathlib import Path

from qgf.downstream.gnns.benchmark import MUTAG_SEEDS, run_trial
from qgf.downstream.gnns.model import ALL_MODELS
from qgf.downstream.gt.benchmark import N_FOLDS

CONFIGS = [
    "rrwp_lap_ring_geom",
    "rrwp_lap_ring_geom_af_fm",
    "rrwp_lap_ring_geom_exact",
    "rrwp_lap_ring_geom_exact_nn",
]
SUBSET = "exact173"


def trial_dir(out: Path, model: str, config: str, fold: int, seed: int) -> Path:
    return out / model / config / f"fold_{fold}" / f"seed_{seed}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    p.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    p.add_argument("--seeds", type=int, nargs="+", default=MUTAG_SEEDS)
    p.add_argument("--out", type=Path, default=Path("results/non_unit_disk/mutag"))
    p.add_argument("--chunk", default="0/1", help="i/n: every n-th trial starting at i")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args(argv)

    trials = list(itertools.product(args.models, args.configs, args.folds, args.seeds))
    i, n = (int(v) for v in args.chunk.split("/"))
    kw = {} if args.data_root is None else {"root": args.data_root}
    failed = 0
    for model, config, fold, seed in trials[i::n]:
        out_dir = trial_dir(args.out, model, config, fold, seed)
        if (out_dir / "results.json").exists():
            continue
        print(f"=== {model} {config} fold={fold} seed={seed}", flush=True)
        try:
            r = run_trial("mutag", model, config, seed, fold, out_dir,
                          max_epochs=args.max_epochs, subset=SUBSET, **kw)  # fmt: skip
            print(f"=== done test_f1_macro={r['test_f1_macro']:.4f}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
