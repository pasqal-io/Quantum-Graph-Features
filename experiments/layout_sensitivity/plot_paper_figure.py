"""Paper figure (appendix, Table 23): placement sensitivity of the AFM correlations.

Means and stds are read from the stored-study summary; the 40 individual cases per level
are recomputed from the raw cases with the functions of `analyze.py` and checked against
the summary and Table 23.

    uv run python experiments/layout_sensitivity/plot_paper_figure.py
"""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from analyze import MODES, load_cases, pooled_spearman  # noqa: E402  (sibling script)
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)  # 0.35 and 0.5 are not in Table 23
REEMBED = ("re-embedding", "re-embedding (post-selected)")
# Table 23 of the paper, (mean, std) to 3 decimals.
TABLE_23 = {
    "sigma=0.02": (0.819, 0.073),
    "sigma=0.05": (0.607, 0.099),
    "sigma=0.1": (0.236, 0.201),
    "sigma=0.2": (-0.017, 0.088),
    "re-embedding": (0.233, 0.287),
    "re-embedding (post-selected)": (0.558, 0.247),
}
TABLE_23_FLOOR = -0.001
QPU_JITTER_PCT = (2.0, 4.0)  # 0.1-0.2 um over d_min = 5 um

AFM = "#2a78d6"  # paper convention (src/qgf/reconstruction/report.py)
REEMB = "#008300"  # validated against AFM on white (dataviz validate_palette.js)
INK, INK_2 = "#1a1a1a", "#52514e"
RULE, GRID, BAND, FLOOR = "#9a9a96", "#e6e6e3", "#eeeeec", "#7a7a76"


def per_case_spearman(run_dir: Path) -> dict[str, np.ndarray]:
    """Pooled Spearman of every case vs. its graph's reference, keyed as in summary.json."""
    cases = load_cases(run_dir)
    is_ref = [c["mode"] == "noise" and c["sigma_rel"] == 0 for c in cases]
    refs = {c["graph_idx"]: c["corr"] for c, r in zip(cases, is_ref) if r}
    out = defaultdict(list)
    for c, r in zip(cases, is_ref):
        if r:
            continue
        key = f"sigma={c['sigma_rel']:g}" if c["mode"] == "noise" else MODES[c["mode"]]
        out[key].append(pooled_spearman(refs[c["graph_idx"]], c["corr"]))
    return {k: np.array(v) for k, v in out.items()}


def check(values: dict[str, np.ndarray], summary: dict) -> None:
    stats = summary["pooled_spearman"]
    for key, v in values.items():
        s = stats[key]
        assert len(v) == s["n"] == 40, (key, len(v))
        assert abs(v.mean() - s["mean"]) < 1e-9 and abs(v.std() - s["std"]) < 1e-9, key
    for key, (mean, std) in TABLE_23.items():
        assert round(stats[key]["mean"], 3) == mean and round(stats[key]["std"], 3) == std, key
    assert round(summary["permutation_floor"], 3) == TABLE_23_FLOOR


def style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{times}",  # NeurIPS: Times text, CM math
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.5,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def summary_marks(ax, x, stats, color, marker, size) -> None:
    """Mean +- std (n = 40) on top of the individual cases."""
    ax.errorbar(
        x,
        [s["mean"] for s in stats],
        yerr=[s["std"] for s in stats],
        fmt=marker,
        ms=size,
        color=color,
        mec="white",
        mew=0.7,
        elinewidth=0.9,
        capsize=1.8,
        capthick=0.9,
        zorder=4,
    )


def plot(values, summary, out: Path) -> None:
    stats = summary["pooled_spearman"]
    floor = summary["permutation_floor"]
    rng = np.random.default_rng(0)  # jitter only
    style()
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.15),
        sharey=True,
        gridspec_kw={"width_ratios": [3.4, 1.1]},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)

    # (a) Gaussian placement noise, sigma in % of d_min on a log axis.
    pct = np.array(SIGMAS) * 100
    ax.axvspan(*QPU_JITTER_PCT, color=BAND, lw=0, zorder=0)
    ax.text(
        np.sqrt(np.prod(QPU_JITTER_PCT)),
        1.01,
        r"QPU jitter",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7,
        color=INK_2,
    )
    for p, s in zip(pct, SIGMAS):
        v = values[f"sigma={s:g}"]
        x = p * 10 ** rng.uniform(-0.03, 0.03, len(v))
        ax.scatter(x, v, s=3, color=AFM, alpha=0.28, lw=0, zorder=2)
    level = [stats[f"sigma={s:g}"] for s in SIGMAS]
    ax.plot(pct, [s["mean"] for s in level], color=AFM, lw=1.0, zorder=3)
    summary_marks(ax, pct, level, AFM, "o", 4.2)
    ax.set_xscale("log")
    ax.set_xlim(1.6, 62)
    ax.set_xticks(pct, [f"{p:g}" for p in pct])
    ax.minorticks_off()
    ax.set_xlabel(r"Placement noise $\sigma$ (\% of $d_{\min}$)")
    ax.set_ylabel(r"Spearman $\rho_s$ vs.\ reference")

    # (b) Re-embeddings: categorical, not on the sigma axis.
    for i, key in enumerate(REEMBED):
        v = values[key]
        bx.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=3, color=REEMB, alpha=0.28, lw=0)
    summary_marks(bx, [0, 1], [stats[k] for k in REEMBED], REEMB, "s", 3.8)
    bx.set_xlim(-0.6, 1.6)
    bx.set_xticks([0, 1], ["raw", "quality-\nselected"])
    bx.set_xlabel("Re-embedding")
    bx.tick_params(axis="y", left=False)
    bx.spines["left"].set_visible(False)

    for a in (ax, bx):
        a.axhline(floor, color=FLOOR, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        a.grid(axis="y", color=GRID, lw=0.5)
        a.set_axisbelow(True)
    ax.text(1.7, floor + 0.03, "random node permutation", fontsize=7, color=INK_2, va="bottom")
    ax.set_ylim(-0.22, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    key = [
        Line2D([], [], marker="o", ls="", ms=4.2, color=INK_2, mec="white", mew=0.7),
        Line2D([], [], marker="o", ls="", ms=1.8, color=INK_2, alpha=0.5, mew=0),
    ]
    ax.legend(
        key,
        [r"mean $\pm$ std ($n = 40$: 4 graphs $\times$ 10 runs)", "single run"],
        loc="upper right",
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    for a, label in ((ax, "(a)"), (bx, "(b)")):
        a.set_title(label, loc="left", fontsize=8, pad=3, color=INK)

    fig.align_xlabels()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    buf = io.BytesIO()  # opaque RGB PNG, same physical size as the PDF
    fig.savefig(buf, format="png", dpi=600)
    Image.open(buf).convert("RGB").save(out.with_suffix(".png"), dpi=(600, 600))
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/layout_sensitivity"))
    parser.add_argument("--summary", type=Path, default=HERE / "expected/summary.json")
    parser.add_argument("--out", type=Path, default=Path("results/paper_figures/layout_noise"))
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    values = per_case_spearman(args.raw)
    check(values, summary)
    plot(values, summary, args.out)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
