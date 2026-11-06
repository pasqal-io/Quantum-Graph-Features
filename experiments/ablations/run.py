"""ZINC_tiny ablation (Tables 8-10): run the trials behind the paper's cells.

The trial list is ``expected/trials.csv`` (one row per original trial, with its training
setting); every distinct (model, dataset, config, size, zinc_mode, setting, seed) is run
once. Each trial writes
``<out>/<model>/<dataset>/<config>/<size>/<zinc_mode>/<setting>/seed_<s>/results.json``;
finished trials are skipped (``--chunk i/n`` splits the list across jobs).

    uv run python experiments/ablations/run.py --models GCN --sizes tiny --limit 1  # one trial
    uv run python experiments/ablations/run.py --chunk 3/64                          # Tables 8-10
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import pandas as pd

from qgf.downstream.gnns.ablation import run_trial

EXPECTED = Path(__file__).parent / "expected"
KEY = ["model_type", "dataset", "config", "size", "zinc_mode", "setting", "seed"]


def trial_dir(out: Path, t) -> Path:
    return (
        out / t.model_type / t.dataset / t.config / t.size / t.zinc_mode / t.setting
        / f"seed_{t.seed}"
    )  # fmt: skip


def trial_list(zinc_modes: list[str], models: list[str] | None, sizes: list[str] | None):
    t = pd.read_csv(EXPECTED / "trials.csv").drop_duplicates(KEY)
    t = t[t.zinc_mode.isin(zinc_modes)]
    if models:
        t = t[t.model_type.isin(models)]
    if sizes:
        t = t[t["size"].isin(sizes)]
    return t.sort_values(KEY).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--zinc-modes", nargs="+", default=["zinc"], help="paper tables: zinc")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--sizes", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("results/ablations"))
    p.add_argument("--chunk", default="0/1", help="i/n: run every n-th trial starting at i")
    p.add_argument("--limit", type=int, default=None, help="run at most this many trials")
    p.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args(argv)
    trials = trial_list(args.zinc_modes, args.models, args.sizes)
    i, n = (int(v) for v in os.path.expandvars(args.chunk).split("/"))
    todo = list(trials.iloc[i::n].itertuples(index=False))[: args.limit]
    kw = {} if args.data_root is None else {"root": args.data_root}
    failed = 0
    for t in todo:
        out_dir = trial_dir(args.out, t)
        if (out_dir / "results.json").exists():
            continue
        print(f"=== {' '.join(str(getattr(t, k)) for k in KEY)}", flush=True)
        try:
            r = run_trial(t.model_type, t.dataset, t.config, t.size, t.zinc_mode, t.setting,
                          int(t.seed), out_dir, max_epochs=args.max_epochs, **kw)  # fmt: skip
            print(f"=== done test_mae={r['test_mae']:.4f}", flush=True)
        except Exception:  # keep going with the other trials of the chunk
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
