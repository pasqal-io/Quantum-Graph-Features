"""Paper figure for Table 18 (App. H): explained variance R^2 of the off-diagonal MUTAG
correlations along the nine cumulative classical descriptor stacks.

Random forest: graph-level 5-fold held-out R^2 at t = 1536 ns, K = 21, band = +-1 fold std.
OLS: in-sample R^2, K = 21, averaged over the 20 times (column `r2_time_avg`, the series
drawn by `qgf.reconstruction.report.plot_ceiling_evolution`).

Reads `expected/ceiling_summary_mutag.csv` and writes
`results/paper_figures/reconstruction_ceiling.{pdf,png}` (5.5 in wide, PNG at 600 dpi).

    uv run python experiments/reconstruction/plot_paper_ceiling.py

Other fold schemes: `--summary <csv>` (same schema; skips the Table 18 value checks and fits
the y-range to the data), `--label <folds>` (legend text replacing "graph-level 5-fold"),
`--out <stem>` (output stem).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from qgf.reconstruction.registry import CUMULATIVE_FAMILIES  # noqa: E402
from qgf.reconstruction.report import AFM_COLOR, FM_COLOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "experiments/reconstruction/expected/ceiling_summary_mutag.csv"
OUT = ROOT / "results/paper_figures/reconstruction_ceiling"

# One label per cumulative step (CUMULATIVE_FAMILIES order). `meanfield` is the static
# susceptibility chi = (1 - beta J)^-1 of the register couplings, not a quench simulation.
STEP_LABELS: tuple[str, ...] = (
    "Walk",
    "+ RRWP",
    "+ Laplacian",
    "+ ring",
    "+ heat,\ngeo., res.",
    r"+ $r_{ij}$, $J_{ij}$",
    "+ dTWA",
    "+ two-atom",
    "+ mean-\nfield $\\chi$",
)
# Groups of steps = rows of Table 18 (each row is the last step of its group).
GROUPS: tuple[tuple[int, int, str], ...] = (
    (0, 2, r"Sec.~3.2 basis"),
    (3, 4, "graph descriptors"),
    (5, 5, "register\ngeometry"),
    (6, 8, "physical controls"),
)
# Paper values (R^2, fold std) per Table 18 row, used only to assert the CSV matches.
TABLE18 = {
    "AFM": ((0.915, 0.005), (0.917, 0.005), (0.917, 0.005), (0.917, 0.005)),
    "FM": ((0.848, 0.041), (0.856, 0.036), (0.855, 0.036), (0.860, 0.031)),
}
TABLE18_UNEXPLAINED_PCT = {"AFM": 8.3, "FM": 14.0}
STYLE = {"AFM": (AFM_COLOR, "o", 4.2), "FM": (FM_COLOR, "s", 3.7)}  # colour, marker, size

INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e6e6e6"
FIG_W, FIG_H = 5.5, 2.5  # in; NeurIPS text width
AXES = (0.085, 0.245, 0.79, 0.605)  # left, bottom, width, height (figure fraction)


def load(csv: Path = CSV, check_table18: bool = True) -> dict[tuple[str, str], pd.DataFrame]:
    """Nine-step series per (probe, quench); RF checked against Table 18 to rounding."""
    frame = pd.read_csv(csv).sort_values("step")
    data = {}
    for (probe, quench), sub in frame.groupby(["probe", "quench"]):
        assert list(sub["family"]) == list(CUMULATIVE_FAMILIES), (probe, quench)
        data[probe, quench] = sub.reset_index(drop=True)
    for quench, paper in TABLE18.items() if check_table18 else ():
        rf = data["random_forest", quench]
        rows = rf.loc[[end for _, end, _ in GROUPS]]
        for (r2, std), (_, row) in zip(paper, rows.iterrows()):
            assert abs(row["r2_ref"] - r2) <= 5e-4 + 1e-9, (quench, row["family"], r2)
            assert abs(row["fold_std_ref"] - std) <= 5e-4 + 1e-9, (quench, row["family"], std)
        final = 100 * rf["unexplained_ref"].iloc[-1]
        assert abs(final - TABLE18_UNEXPLAINED_PCT[quench]) <= 0.05 + 1e-9, (quench, final)
    return data


def y_range(data: dict[tuple[str, str], pd.DataFrame], step: float = 0.05) -> tuple[float, float]:
    """Smallest `step`-aligned range holding the OLS lines and the RF bands."""
    drawn = [data["ols", q]["r2_time_avg"] for q in STYLE]
    for q in STYLE:
        rf = data["random_forest", q]
        drawn += [rf["r2_ref"] - rf["fold_std_ref"], rf["r2_ref"] + rf["fold_std_ref"]]
    values = pd.concat(drawn)
    lo, hi = math.floor(values.min() / step) * step, math.ceil(values.max() / step) * step
    return round(lo, 2), round(hi, 2)


def plot(
    data: dict[tuple[str, str], pd.DataFrame],
    folds: str = "graph-level 5-fold",
    ylim: tuple[float, float] = (0.70, 0.95),
) -> plt.Figure:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{times}",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = fig.add_axes(AXES)
    steps = range(len(CUMULATIVE_FAMILIES))

    for quench, (color, marker, size) in STYLE.items():
        rf = data["random_forest", quench]
        ols = data["ols", quench]
        ax.fill_between(
            steps,
            rf["r2_ref"] - rf["fold_std_ref"],
            rf["r2_ref"] + rf["fold_std_ref"],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            steps,
            ols["r2_time_avg"],
            color=color,
            linestyle=(0, (3.5, 2)),
            linewidth=0.9,
            marker=marker,
            markersize=size - 0.6,
            markerfacecolor="white",
            markeredgewidth=0.8,
            zorder=2,
            label=rf"{quench}, OLS",
        )
        ax.plot(
            steps,
            rf["r2_ref"],
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=size,
            markeredgecolor="white",  # surface ring where markers sit on other lines
            markeredgewidth=0.6,
            zorder=3,
            label=rf"{quench}, random forest",
        )
        last = rf.iloc[-1]
        ax.annotate(
            rf"{last['r2_ref']:.3f}" + "\n" + rf"({100 * last['unexplained_ref']:.1f}\% unexpl.)",
            (len(steps) - 1, last["r2_ref"]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=INK,
            linespacing=1.1,
            annotation_clip=False,
        )

    ax.set_xlim(-0.4, len(steps) - 0.6)
    ax.set_ylim(*ylim)
    yticks = [round(ylim[0] + 0.05 * i, 2) for i in range(round((ylim[1] - ylim[0]) / 0.05) + 1)]
    ax.set_yticks(yticks, [f"{y:.2f}" for y in yticks])  # text-mode digits (Times)
    ax.set_xticks(list(steps), STEP_LABELS)
    ax.set_ylabel(r"Explained variance $R^2$", color=INK, labelpad=3)
    ax.tick_params(colors=MUTED, labelcolor=INK, pad=2)
    ax.tick_params(axis="x", length=0, pad=3)
    for label in ax.get_xticklabels():
        label.set_linespacing(1.0)
        label.set_verticalalignment("top")
        label.set_multialignment("center")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    # Hairline separators between the Table 18 groups.
    for _, end, _ in GROUPS[:-1]:
        ax.axvline(end + 0.5, color=GRID, linewidth=0.5, zorder=0)

    # Legend as a 2 x 3 grid: rows = probe (line style), columns = quench, then probe definition.
    handles, _ = ax.get_legend_handles_labels()  # AFM OLS, AFM RF, FM OLS, FM RF
    blank = Line2D([], [], linestyle="none")
    legend = ax.legend(
        [handles[1], handles[0], handles[3], handles[2], blank, blank],
        [
            "AFM",
            "AFM",
            "FM",
            "FM",
            rf"Random forest: held-out, {folds}, $t = 1536$\,ns, band $\pm1$ s.d.",
            "OLS: in-sample, averaged over the 20 times",
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handlelength=2.6,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.35,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    for text in legend.get_texts()[4:]:
        text.set_color(MUTED)

    # Brackets under the step labels: one per Table 18 row.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    low = min(t.get_window_extent(renderer).y0 for t in ax.get_xticklabels())
    y = ax.transAxes.inverted().transform((0, low - 4))[1]  # 4 px at 100 dpi below labels
    tick = 0.03
    for start, end, name in GROUPS:
        xs = (start - 0.42, end + 0.42)
        ax.plot(
            [xs[0], xs[0], xs[1], xs[1]],
            [y + tick, y, y, y + tick],
            transform=ax.get_xaxis_transform(),
            color=MUTED,
            linewidth=0.6,
            clip_on=False,
        )
        ax.text(
            (start + end) / 2,
            y - 0.025,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color=MUTED,
            linespacing=1.0,
        )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper figure for Table 18 (App. H).")
    parser.add_argument("--summary", type=Path, help="summary CSV, default schema")
    parser.add_argument("--label", default="graph-level 5-fold", help="RF folds, in the legend")
    parser.add_argument("--out", type=Path, default=OUT, help="output stem (.pdf and .png)")
    args = parser.parse_args()
    if args.summary is None:
        fig = plot(load(), args.label)
    else:
        data = load(args.summary, check_table18=False)
        fig = plot(data, args.label, y_range(data))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": both files keep the 5.5 in physical width.
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    print(out.with_suffix(".pdf"), out.with_suffix(".png"), sep="\n")


if __name__ == "__main__":
    main()
