"""PCA of the temporal quantum features (App. D.2, Fig. 4).

Per pulse (AFM, FM) and feature type, features are z-scored per time step over the
whole dataset (as for training, "per_step"), then rows are collected: one per node
(one-body, [N_total, 20]) and one per directed edge (two-body, [E_total, 20]); PCA is
fit on each matrix.

    uv run python experiments/temporal_pca/run.py --dataset zinc   # Fig. 4
    uv run python experiments/temporal_pca/run.py --dataset mutag
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from qgf.data import load, load_splits

N_PC_SHOW = 10
EPS = 1e-6


class PCAResult(NamedTuple):
    var_ratio: np.ndarray  # [K]
    loadings: np.ndarray  # [K, K] rows = PCs


def load_all(dataset: str, pulse: str) -> list:
    if dataset == "mutag":
        return load(f"mutag_{pulse}")[0]
    splits, _ = load_splits(f"zinc_{pulse}")
    return splits["train"] + splits["val"] + splits["test"]


def per_step_stats(graphs) -> tuple:
    """Mean/std per time step over all nodes (one-body) and all N*N pairs (two-body)."""
    ob = torch.cat([g.onebody_features.T for g in graphs])  # [sum N, T]
    tb = torch.cat(
        [
            g.twobody_features.permute(1, 2, 0).reshape(-1, g.twobody_features.shape[0])
            for g in graphs
        ]
    )
    return ob.mean(0), ob.std(0) + EPS, tb.mean(0), tb.std(0) + EPS


def collect_features(graphs) -> tuple[np.ndarray, np.ndarray]:
    """``(one-body [N_total, T], two-body [E_total, T])`` of the normalised features."""
    ob_mean, ob_std, tb_mean, tb_std = per_step_stats(graphs)
    ob_list, tb_list = [], []
    for g in graphs:
        ob_list.append(((g.onebody_features.T - ob_mean) / ob_std).numpy())
        tb = (g.twobody_features.permute(1, 2, 0) - tb_mean) / tb_std  # [N, N, T]
        ei = g.edge_index
        tb_list.append(tb[ei[0], ei[1], :].numpy())
    return np.concatenate(ob_list), np.concatenate(tb_list)


def fit_pca(matrix: np.ndarray) -> PCAResult:
    pca = PCA(n_components=matrix.shape[1]).fit(matrix)
    return PCAResult(pca.explained_variance_ratio_, pca.components_)


def summary(result: PCAResult, shape) -> dict:
    cum = np.cumsum(result.var_ratio)
    return {
        "shape": list(shape),
        "var_ratio": result.var_ratio.round(6).tolist(),
        "n_components": {
            f"{int(t * 100)}%": int(np.searchsorted(cum, t)) + 1 for t in (0.8, 0.9, 0.95, 0.99)
        },
    }


def _scree(ax, var_ratio, label, title):
    k = len(var_ratio)
    x = np.arange(1, k + 1)
    ax.bar(x, var_ratio * 100, color="C4", zorder=3)
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(var_ratio) * 100, color="C2", marker="o", ms=2.5)
    for y, ls in ((90, "--"), (95, ":")):
        ax2.axhline(y, color="C2", lw=0.5, ls=ls, alpha=0.5)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cumulative (%)", color="C2")
    ax.set_xticks(x[:: 2 if k > 10 else 1])
    ax.set_xlabel(r"Principal component $k$")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(title, fontsize=8, fontstyle="italic")
    ax.set_title(label, loc="left", fontsize=9, fontweight="bold")


def _loadings(ax, loadings, label, title):
    n_show = min(N_PC_SHOW, loadings.shape[0])
    mat = loadings[:n_show].T  # [T, PCs]
    vmax = float(np.abs(mat).max())
    im = ax.imshow(mat, aspect="auto", cmap="PuOr", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(n_show), [str(i + 1) for i in range(n_show)])
    k = mat.shape[0]
    ax.set_yticks(range(0, k, 2), [str(t + 1) for t in range(0, k, 2)])
    ax.set_xlabel(r"Principal component $k$")
    ax.set_ylabel(r"Timestep $t$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Loading")
    ax.set_title(title, fontsize=8, fontstyle="italic")
    ax.set_title(label, loc="left", fontsize=9, fontweight="bold")


def plot(results: dict, title: str, out: Path) -> None:
    rows = [
        ("afm", "ob", "AFM — 1-body"),
        ("fm", "ob", "FM — 1-body"),
        ("afm", "tb", "AFM — 2-body"),
        ("fm", "tb", "FM — 2-body"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 11.0))
    for r, (pulse, kind, name) in enumerate(rows):
        res = results[pulse][kind]
        _scree(axes[r, 0], res.var_ratio, f"({'aceg'[r]})", name)
        _loadings(axes[r, 1], res.loadings, f"({'bdfh'[r]})", name)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.05, top=0.95, wspace=0.45, hspace=0.45)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["mutag", "zinc"], default="zinc")
    parser.add_argument("--out", type=Path, default=Path("results/temporal_pca"))
    args = parser.parse_args(argv)
    results, report = {}, {}
    for pulse in ("afm", "fm"):
        ob, tb = collect_features(load_all(args.dataset, pulse))
        results[pulse] = {"ob": fit_pca(ob), "tb": fit_pca(tb)}
        report[pulse] = {
            "onebody": summary(results[pulse]["ob"], ob.shape),
            "twobody": summary(results[pulse]["tb"], tb.shape),
        }
        for kind in ("onebody", "twobody"):
            r = report[pulse][kind]
            print(
                f"{args.dataset} {pulse} {kind} {r['shape']}: PC1 {100 * r['var_ratio'][0]:.1f}%, "
                f"PC2 {100 * r['var_ratio'][1]:.1f}%, PC3 {100 * r['var_ratio'][2]:.1f}%, "
                f"components {r['n_components']}",
                flush=True,
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"pca_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    name = {"zinc": r"$\mathrm{ZINC}_{\mathrm{subset}}^{*}$", "mutag": "MUTAG"}[args.dataset]
    plot(results, f"PCA of quantum features — {name}", args.out / f"pca_{args.dataset}.png")
    print(f"wrote {args.out}/pca_{args.dataset}.json/.png")


if __name__ == "__main__":
    main()
