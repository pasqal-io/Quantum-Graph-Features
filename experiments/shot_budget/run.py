"""Shot budget for the hardware run (App. D.3).

1. CLT + Bonferroni budget ``shots_clt(N)`` over the ``M = N(N+1)/2`` connected
   correlations (v_max = 0.06, eps = 0.03, delta = 0.05), for every MUTAG size.
2. Empirical per-shot variance of the connected-correlation estimators, by bootstrap
   over the emulated shots (200 resamples, max over pairs and times per graph): the
   ``v_max`` input of the formula. Port of the original notebook; with ``--epsilon 0.01``
   on FM it reproduces its printed per-graph budgets.

    uv run python experiments/shot_budget/run.py --raw-dir data/raw
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from qgf.generate.qpu import shots_clt
from qgf.generate.raw import load_raw, raw_path


def n_shots_clt(sigma2: float, eps: float, delta: float, m: int) -> int:
    k = norm.ppf(1 - delta / (2 * m))
    return math.ceil(k**2 * sigma2 / eps**2)


def n_shots_bernstein(sigma2: float, r: float, eps: float, delta: float, m: int) -> int:
    return math.ceil((2 * sigma2 / eps**2 + 2 * r / (3 * eps)) * math.log(2 * m / delta))


def bootstrap_correlation_var(raw_shots: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Per-shot variance of every C_ij (i <= j) estimator, [n_pairs]; vectorised over pairs,
    same random draws as the original pair loop."""
    n_shots, n = raw_shots.shape
    iu = np.triu_indices(n)
    x = raw_shots.astype(np.float64)
    boot = np.empty((n_boot, len(iu[0])))
    for b in range(n_boot):
        s = x[rng.integers(0, n_shots, size=n_shots)]
        p = s.mean(axis=0)
        boot[b] = ((s.T @ s) / n_shots - np.outer(p, p))[iu]
    return n_shots * boot.var(axis=0, ddof=1)


def expand(raw, g: int, t: int) -> np.ndarray:
    bits, w = raw.bit_matrix(g, t)
    return np.repeat(bits, w.astype(int), axis=0)


def correlation_budgets(raw, eps: float, delta: float, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for g in range(len(raw)):
        n = int(raw.num_nodes[g])
        m = n * (n + 1) // 2
        sigma2 = max(
            float(bootstrap_correlation_var(expand(raw, g, t), n_boot, rng).max())
            for t in range(len(raw.times_ns))
        )
        out.append(
            {
                "graph": int(raw.initial_id[g]),
                "N": n,
                "M": m,
                "sigma2_max": sigma2,
                "clt": n_shots_clt(sigma2, eps, delta, m),
                "bernstein": n_shots_bernstein(sigma2, 0.5, eps, delta, m),
            }
        )
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--pulses", nargs="*", default=["afm", "fm"])
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None, help="smoke runs")
    parser.add_argument("--out", type=Path, default=Path("results/shot_budget"))
    args = parser.parse_args(argv)

    sizes = range(10, 29)
    table = [
        {
            "N": n,
            "M": n * (n + 1) // 2,
            "k_N": float(norm.ppf(1 - 0.05 / (n * (n + 1)))),
            "shots": shots_clt(n),
        }
        for n in sizes
    ]
    print("N  M    k_N    shots_clt")
    for r in table:
        print(f"{r['N']:2d} {r['M']:4d} {r['k_N']:.4f} {r['shots']}")
    report = {"formula": table, "empirical": {}}
    for pulse in args.pulses:
        raw = load_raw(raw_path(args.raw_dir, f"mutag_{pulse}"))
        if args.max_graphs:
            raw.initial_id = raw.initial_id[: args.max_graphs]
        b = correlation_budgets(raw, args.epsilon, args.delta, args.n_boot, args.seed)
        s = np.array([r["sigma2_max"] for r in b])
        report["empirical"][pulse] = {
            "epsilon": args.epsilon,
            "sigma2_max": {
                "median": float(np.median(s)),
                "max": float(s.max()),
                "min": float(s.min()),
            },
            "graphs": b,
        }
        print(
            f"{pulse}: per-graph max per-shot variance of C_ij: median {np.median(s):.4f}, "
            f"range [{s.min():.4f}, {s.max():.4f}]",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shot_budget.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}/shot_budget.json")


if __name__ == "__main__":
    main()
