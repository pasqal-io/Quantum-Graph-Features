"""Collect the per-trial results of the original ESA MUTAG runs (paper Table 4).

Internal use only: reads the result tree of the original runs (path given as argument);
not needed to reproduce results from the released data.

Sources: ``esa_mutag_5seeds`` (emulated features, concatenation and gated fusion) and
``esa_mutag_afm_qpu`` (hardware AFM, configs ``afm_qpu*`` renamed ``hw_afm_3t*``: these runs
used the hardware set at 305, 512 and 4000 ns, i.e. the released ``mutag_hw_afm`` without its
715 ns run, in the ZZ convention and clipped to [-1, 1]). Metric: binary F1
(``test_f1``, positive class) as stored by the original runs.

    python scripts/internal/esa_reference.py --results <esa repo>/results \
        --out experiments/esa/expected
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RUNS = ["esa_mutag_5seeds", "esa_mutag_afm_qpu"]
RENAME = {"afm_qpu": "hw_afm_3t", "afm_qpu_stat": "hw_afm_3t_stat"}


def collect(results: Path) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for p in sorted((results / run).glob("*/*/fold_*/seed_*/results.json")):
            r = json.loads(p.read_text())
            rows.append(
                {
                    "fusion": r["fusion_mode"],
                    "config": RENAME.get(r["config"], r["config"]),
                    "fold": r["fold"],
                    "seed": r["seed"],
                    "f1": r["test_f1"],
                    "accuracy": r["test_accuracy"],
                    "auc": r["test_auc"],
                    "mcc": r["test_mcc"],
                    "gate_mean": r.get("gate_mean"),
                    "abs_pe_dim": r["abs_pe_dim"],
                    "source": run,
                }
            )
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby(["fusion", "config"], sort=False)
    out = g["f1"].agg(mean="mean", std="std", n_trials="size")
    out["std_ddof0"] = g["f1"].std(ddof=0)
    out["gate_mean"] = g["gate_mean"].mean()
    out["abs_pe_dim"] = g["abs_pe_dim"].first()
    out["source"] = g["source"].first()
    return out.reset_index()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    trials = collect(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out / "mutag_trials.csv", index=False, float_format="%.6f")
    summarize(trials).to_csv(args.out / "mutag_summary.csv", index=False, float_format="%.6f")
    print(f"{len(trials)} trials, {trials.groupby(['fusion', 'config']).ngroups} cells")


if __name__ == "__main__":
    main()
