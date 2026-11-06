"""GT capacity study on ZINC*_subset (paper Fig. 6): run trials, or plot and summarise.

Three feature sets (RRWP 20/20, RRWP + AFM_stat 24/24, RRWP + AFM 40/40) at three model
widths (hidden 64 / 128 / 256 with 4 / 8 / 16 heads and FFN 128 / 256 / 512; 4 layers),
seeds 0-4. Training as the GT ZINC benchmark (``qgf.downstream.gt.benchmark``): statistics
and training on the training split, early stopping and learning-rate plateau on the
validation split, test MAE / RMSE reported once.

    uv run python experiments/ablations/capacity.py run --hidden-dims 64 --seeds 0 --max-epochs 2
    sbatch experiments/ablations/submit_capacity.sh                        # 45 trials, GPU
    uv run python experiments/ablations/capacity.py plot --results results/ablations/capacity \\
        --out results/ablations/capacity
    uv run python experiments/ablations/capacity.py plot \\
        --out experiments/ablations/expected/capacity
"""

from __future__ import annotations

import argparse
import itertools
import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qgf.downstream.gt.benchmark import run_trial  # noqa: E402

EXPECTED = Path(__file__).parent / "expected"
CONFIGS = {"rrwp": "RRWP\n(20, 20)", "rrwp_af_stat": "RRWP + AFM$_{stat}$\n(24, 24)",
           "rrwp_af": "RRWP + AFM\n(40, 40)"}  # fmt: skip
CAPACITIES = {
    64: {"hidden_dim": 64, "num_heads": 4, "ffn_dim": 128},
    128: {"hidden_dim": 128, "num_heads": 8, "ffn_dim": 256},
    256: {"hidden_dim": 256, "num_heads": 16, "ffn_dim": 512},
}
SEEDS = [0, 1, 2, 3, 4]


def trial_dir(out: Path, config: str, hidden: int, seed: int) -> Path:
    return out / config / f"hidden_{hidden}" / f"seed_{seed}"


def run(args) -> int:
    trials = list(itertools.product(args.configs, args.hidden_dims, args.seeds))
    i, n = (int(v) for v in args.chunk.split("/"))
    kw = {} if args.data_root is None else {"root": args.data_root}
    failed = 0
    for config, hidden, seed in trials[i::n]:
        out_dir = trial_dir(args.out, config, hidden, seed)
        if (out_dir / "results.json").exists():
            continue
        print(f"=== {config} hidden={hidden} seed={seed}", flush=True)
        try:
            r = run_trial("zinc", config, seed, None, out_dir, max_epochs=args.max_epochs,
                          model_overrides=CAPACITIES[hidden], **kw)  # fmt: skip
            print(f"=== done test_mae={r['test_mae']:.4f} params={r['num_params']}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


def read_results(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/hidden_*/seed_*/results.json")):
        r = json.loads(f.read_text())
        rows.append({"config": r["config"], "hidden_dim": r["model_config"]["hidden_dim"],
                     "seed": r["seed"], "test_mae": r["test_mae"], "test_rmse": r["test_rmse"],
                     "num_params": r["num_params"], "best_epoch": r.get("best_epoch")})  # fmt: skip
    return pd.DataFrame(rows)


def summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Mean and std (ddof 0, as the original figure) per (config, hidden_dim)."""
    g = trials.groupby(["config", "hidden_dim"])
    out = g.agg(
        mae_mean=("test_mae", "mean"),
        mae_std=("test_mae", lambda v: np.std(v)),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", lambda v: np.std(v)),
        n=("seed", "count"),
    )
    if "num_params" in trials:
        out["num_params"] = g.num_params.first()
    return out.reset_index()


def figure(summ: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    width = 0.25
    x = np.arange(len(CONFIGS))
    for ax, metric, label in ((axes[0], "mae", "Test MAE"), (axes[1], "rmse", "Test RMSE")):
        for k, hidden in enumerate(CAPACITIES):
            s = summ[summ.hidden_dim == hidden].set_index("config").reindex(list(CONFIGS))
            params = s["num_params"].iloc[0] if "num_params" in s else None
            name = f"D = {hidden}" + (f" (~{params / 1e6:.2f}M)" if params else "")
            ax.bar(x + (k - 1) * width, s[f"{metric}_mean"], width, yerr=s[f"{metric}_std"],
                   capsize=2, label=name)  # fmt: skip
        ax.set_xticks(x, list(CONFIGS.values()), fontsize=7)
        ax.set_xlabel("Feature config")
        ax.set_ylabel(label)
        lo = (summ[f"{metric}_mean"] - summ[f"{metric}_std"]).min()
        hi = (summ[f"{metric}_mean"] + summ[f"{metric}_std"]).max()
        ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.set_title(f"({'ab'[metric == 'rmse']}) {label}", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot(args) -> int:
    if args.results is None:
        trials, title = pd.read_csv(EXPECTED / "capacity_trials.csv"), "GT, ZINC*_subset (original)"
    else:
        trials, title = read_results(args.results), "GT, ZINC*_subset (re-run)"
    summ = summary(trials)
    args.out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out / "capacity_summary.csv", index=False, float_format="%.6f")
    figure(summ, args.out / "fig6_capacity.pdf", title)
    print(summ.to_string(index=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    r.add_argument("--hidden-dims", type=int, nargs="+", default=list(CAPACITIES))
    r.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    r.add_argument("--out", type=Path, default=Path("results/ablations/capacity"))
    r.add_argument("--chunk", default="0/1")
    r.add_argument("--max-epochs", type=int, default=None)
    r.add_argument("--data-root", type=Path, default=None)
    q = sub.add_parser("plot")
    q.add_argument("--results", type=Path, default=None, help="re-run tree (default: original)")
    q.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    return run(args) if args.cmd == "run" else plot(args)


if __name__ == "__main__":
    raise SystemExit(main())
