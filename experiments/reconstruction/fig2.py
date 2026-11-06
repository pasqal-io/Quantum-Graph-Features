"""Fig. 2: linear recovery of short-time quantum correlations from adjacency powers / RRWP.

Toy dataset: the two MUTAG graphs with N = 10 (indices 75 and 115). For each quench
(AFM, FM) the connected correlations ``C_ij(t)`` of the graph Hamiltonian
``H = sum_{i<j} J_NN A_ij n_i n_j + sum_i [Omega/2 sx_i - delta n_i]`` are computed by
exact diagonalisation from ``|0...0>`` on 20 times in [10, 400] ns. At each time, one
least-squares fit pooled over the 90 pairs ``i < j`` of both graphs predicts ``C_ij(t)``
from ``[1, (A^k)_ij]_{k <= K}`` (adjacency powers) or ``[1, (M^k)_ij]_{k <= K}``
(RRWP, ``M = D^-1 A``); the curve is the in-sample R^2.

Parameters as in the published figure: ``J_NN = 54.75`` rad/us; AFM ``(Omega, delta) =
(0.3, 0) J_NN``; FM ``(0.5, 0.5) J_NN``, i.e. ``delta = delta_inf`` for these graphs
(``|E| = N``). The paper's FM rule ``delta = 2 delta_inf`` gives ``delta = J_NN`` here:
``--fm-delta paper`` uses it.

    uv run python experiments/reconstruction/fig2.py --out results/reconstruction/fig2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qgf.data import DEFAULT_ROOT, load  # noqa: E402
from qgf.reconstruction.quench import quench_correlations  # noqa: E402

J_NN = 54.75  # rad/us, value of the original run (C6 / r_NN^6 = 54.76)
TIMES_NS = np.linspace(10.0, 400.0, 20)
K_VALUES = [1, 3, 5, 7, 10, 14, 21, 30, 50]
K_SHOWN = [1, 3, 7, 21]
NUM_NODES = 10
QUENCHES = {  # (Omega, delta) / J_NN
    "published": {"AFM": (0.3, 0.0), "FM": (0.5, 0.5)},
    "paper": {"AFM": (0.3, 0.0), "FM": (0.5, 1.0)},
}


def toy_graphs(root: Path = DEFAULT_ROOT) -> tuple[list[int], list[np.ndarray]]:
    """Indices and adjacency matrices of the MUTAG graphs with N = 10."""
    graphs, _ = load("mutag_afm", root=root)
    ids, adj = [], []
    for k, g in enumerate(graphs):
        if g.num_nodes == NUM_NODES:
            a = np.zeros((NUM_NODES, NUM_NODES))
            a[g.edge_index[0].numpy(), g.edge_index[1].numpy()] = 1.0
            ids.append(k)
            adj.append(np.maximum(a, a.T))
    return ids, adj


def upper_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def power_features(matrix: np.ndarray, k_max: int, start_identity: bool) -> np.ndarray:
    """``[1, (P^1)_ij, ..., (P^K)_ij]`` over the pairs ``i < j`` (P = A or M)."""
    i, j = upper_pairs(matrix.shape[0])
    cols = [np.ones(len(i))]
    cur = np.eye(matrix.shape[0]) if start_identity else None
    for _ in range(k_max):
        cur = matrix.copy() if cur is None else cur @ matrix
        cols.append(cur[i, j])
    return np.stack(cols, axis=1)


def walk_features(a: np.ndarray, k_max: int) -> np.ndarray:
    return power_features(a, k_max, start_identity=True)


def rrwp_features(a: np.ndarray, k_max: int) -> np.ndarray:
    deg = a.sum(axis=1)
    m = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)[:, None] * a
    return power_features(m, k_max, start_identity=False)


def pooled_r2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """In-sample R^2 per time of one lstsq fit (columns scaled by max |x|); ``y [T, P]``."""
    scale = np.abs(x).max(axis=0)
    xn = x / np.where(scale > 0, scale, 1.0)
    out = np.empty(len(y))
    for t, yt in enumerate(y):
        coef, *_ = np.linalg.lstsq(xn, yt, rcond=1e-10)
        ss_res = float(((yt - xn @ coef) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        out[t] = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return out


def recovery_curves(adj: list[np.ndarray], quench: str = "published") -> pd.DataFrame:
    """Long table (pulse, family, K, t_ns, r2)."""
    rows = []
    for pulse, (omega, delta) in QUENCHES[quench].items():
        targets = []
        for a in adj:
            c = quench_correlations(J_NN * a, TIMES_NS / 1000.0, omega * J_NN, delta * J_NN)
            i, j = upper_pairs(a.shape[0])
            targets.append(c[:, i, j])  # [T, pairs]
        y = np.concatenate(targets, axis=1)  # [T, 90]
        for family, feats in (("adjacency", walk_features), ("rrwp", rrwp_features)):
            for k in K_VALUES:
                x = np.concatenate([feats(a, k) for a in adj])
                for t, r2 in zip(TIMES_NS, pooled_r2(x, y)):
                    rows.append({"pulse": pulse, "family": family, "K": k, "t_ns": t, "r2": r2})
    return pd.DataFrame(rows)


def figure(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    colors = dict(zip(K_SHOWN, plt.cm.viridis(np.linspace(0.1, 0.85, len(K_SHOWN)))))
    for ax, pulse in zip(axes, ("AFM", "FM")):
        for k in K_SHOWN:
            for family, ls in (("adjacency", "-"), ("rrwp", "--")):
                sel = (curves.pulse == pulse) & (curves.family == family)
                c = curves[sel & curves.K.eq(k)]
                label = f"K = {k}" if family == "adjacency" else None
                ax.plot(c.t_ns, c.r2, ls, color=colors[k], lw=1.5, label=label)
        ax.plot([], [], "k--", label="RRWP")
        ax.set_title(f"({'ab'[pulse == 'FM']}) {pulse}", fontsize=9)
        ax.set_xlabel("time t (ns)")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("pooled $R^2$")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fm-delta", choices=list(QUENCHES), default="published")
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args(argv)
    ids, adj = toy_graphs(args.data_root)
    print(f"MUTAG graphs with N = {NUM_NODES}: {ids}")
    curves = recovery_curves(adj, args.fm_delta)
    args.out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.out / f"fig2_r2_{args.fm_delta}.csv", index=False, float_format="%.8f")
    figure(curves, args.out / f"fig2_recovery_{args.fm_delta}.pdf")
    late = curves[(curves.K.isin(K_SHOWN)) & (curves.t_ns == TIMES_NS[-1])]
    print(late.pivot_table(index=["pulse", "family"], columns="K", values="r2").round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
