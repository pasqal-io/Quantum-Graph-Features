"""Deliverables: serialisation, tables, K x t heatmaps, extension protocol, ceiling figure.

The result table is indexed by (dataset, quench, family, K, t, probe).
"""

from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from qgf.reconstruction.config import REFERENCE_TIME_INDEX
from qgf.reconstruction.registry import BASELINE_FAMILIES, CUMULATIVE_FAMILIES

LOGGER = logging.getLogger(__name__)


def write_results(frame: pd.DataFrame, path: str) -> str:
    """Writes the result table (parquet, or csv/csv.gz depending on the suffix)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet"):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def read_results(path: str) -> pd.DataFrame:
    """Reads a result table, whatever its format."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def table_time_averaged(results: pd.DataFrame, probe: str = "ols") -> pd.DataFrame:
    """Table 13: R^2 averaged over the time grid, rows K, columns family/quench."""
    subset = results[results["probe"] == probe]
    return (
        subset.pivot_table(
            index="k",
            columns=["dataset", "quench", "family"],
            values="r2",
            aggfunc="mean",
        )
        .sort_index()
        .round(4)
    )


def table_at_time(
    results: pd.DataFrame,
    quench: str,
    time_index: int = REFERENCE_TIME_INDEX,
    probes: tuple[str, ...] = ("ols", "kernel_ridge", "random_forest"),
) -> pd.DataFrame:
    """Tables 14/15: R^2 per family and probe at the reference time index."""
    subset = results[
        (results["quench"] == quench)
        & results["probe"].isin(probes)
        & (results["time_index"] == time_index)
    ]
    if subset.empty:
        return subset
    table = subset.pivot_table(
        index=["dataset", "family", "k"],
        columns="probe",
        values="r2",
        aggfunc="mean",
    ).round(4)
    table.attrs["time_ns"] = float(subset["time_ns"].iloc[0])
    return table


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX rendering of a regenerated table."""
    return table.to_latex(
        caption=caption,
        label=label,
        float_format="%.3f",
        na_rep="--",
        multicolumn=True,
        multicolumn_format="c",
    )


def ceiling_summary(results: pd.DataFrame, time_index: int = REFERENCE_TIME_INDEX) -> pd.DataFrame:
    """Cumulative-stack table: R^2 and unexplained variance per (probe, quench, stack).

    Instantaneous probes (random forest) are read at ``time_index``; the OLS is reported
    both at that index and averaged over the time grid.
    """
    rows = []
    order = {family: index for index, family in enumerate(CUMULATIVE_FAMILIES)}
    subset = results[results["family"].isin(order)]
    for (probe, quench, family), group in subset.groupby(["probe", "quench", "family"]):
        at_ref = group[group["time_index"] == time_index]
        rows.append(
            {
                "step": order[family],
                "family": family,
                "probe": probe,
                "quench": quench,
                "r2_ref": float(at_ref["r2"].mean()) if not at_ref.empty else np.nan,
                "r2_time_avg": float(group["r2"].mean()),
                "fold_std_ref": float(at_ref["r2_fold_std"].mean())
                if "r2_fold_std" in at_ref and not at_ref.empty
                else np.nan,
            },
        )
    frame = pd.DataFrame(rows).sort_values(["probe", "quench", "step"]).reset_index(drop=True)
    frame["unexplained_ref"] = 1.0 - frame["r2_ref"]
    return frame


# --------------------------------------------------------------------------------------
# Extension protocol
# --------------------------------------------------------------------------------------


def extension_report(
    results: pd.DataFrame,
    baselines: tuple[str, ...] = BASELINE_FAMILIES,
) -> pd.DataFrame:
    """Solo, incremental and relative residual gain of every extension base."""
    keys = ["dataset", "quench", "probe", "k", "time_index", "time_ns"]
    rows = []
    for baseline in baselines:
        base = results[results["family"] == baseline][keys + ["r2", "num_columns"]]
        base = base.rename(columns={"r2": "r2_baseline", "num_columns": "columns_baseline"})
        prefix = f"{baseline}+"
        combined = results[results["family"].str.startswith(prefix)]
        for family, group in combined.groupby("family"):
            extension = family[len(prefix) :]
            solo = results[results["family"] == extension][keys + ["r2"]].rename(
                columns={"r2": "r2_solo"},
            )
            merged = group.merge(base, on=keys, how="inner").merge(solo, on=keys, how="left")
            merged["baseline"] = baseline
            merged["extension"] = extension
            merged["delta_r2"] = merged["r2"] - merged["r2_baseline"]
            residual = 1.0 - merged["r2_baseline"]
            merged["residual_fraction"] = np.where(
                residual > 0,
                merged["delta_r2"] / residual,
                np.nan,
            )
            merged["extra_columns"] = merged["num_columns"] - merged["columns_baseline"]
            rows.append(
                merged[
                    keys
                    + [
                        "baseline",
                        "extension",
                        "r2_baseline",
                        "r2_solo",
                        "r2",
                        "delta_r2",
                        "residual_fraction",
                        "extra_columns",
                        "r2_adjusted",
                    ]
                ],
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns={"r2": "r2_combined"})


def quench_contrast(results: pd.DataFrame) -> pd.DataFrame:
    """AFM / FM contrast."""
    keys = ["dataset", "family", "probe", "k", "time_index", "time_ns"]
    pivot = results.pivot_table(index=keys, columns="quench", values="r2").reset_index()
    if "AFM" in pivot and "FM" in pivot:
        pivot["afm_minus_fm"] = pivot["AFM"] - pivot["FM"]
        pivot["residual_afm"] = 1.0 - pivot["AFM"]
        pivot["residual_fm"] = 1.0 - pivot["FM"]
    return pivot


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def plot_heatmaps(
    results: pd.DataFrame,
    output_path: str,
    probe: str = "ols",
    vmin: float | None = None,
    vmax: float = 1.0,
) -> str:
    """R^2(K, t) heatmaps, one panel per (quench, family)."""
    subset = results[results["probe"] == probe]
    if subset.empty:
        raise ValueError(f"No result for probe {probe}")
    quenches = sorted(subset["quench"].unique())
    families = sorted(subset["family"].unique())
    figure, axes = plt.subplots(
        len(families),
        len(quenches),
        figsize=(4.2 * len(quenches), 3.0 * len(families)),
        squeeze=False,
    )
    if vmin is None:
        vmin = float(np.nanpercentile(subset["r2"], 2))

    for row, family in enumerate(families):
        for column, quench in enumerate(quenches):
            axis = axes[row][column]
            cell = subset[(subset["family"] == family) & (subset["quench"] == quench)]
            if cell.empty:
                axis.set_axis_off()
                continue
            grid = cell.pivot_table(index="k", columns="time_ns", values="r2", aggfunc="mean")
            image = axis.imshow(
                grid.to_numpy(),
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            axis.set_yticks(range(len(grid.index)))
            axis.set_yticklabels(grid.index)
            ticks = np.linspace(0, len(grid.columns) - 1, min(5, len(grid.columns))).astype(int)
            axis.set_xticks(ticks)
            axis.set_xticklabels([f"{grid.columns[t]:.0f}" for t in ticks])
            axis.set_title(f"{family} - {quench}", fontsize=9)
            axis.set_xlabel("t (ns)")
            axis.set_ylabel("K")
            figure.colorbar(image, ax=axis, label="$R^2$")

    figure.suptitle(f"QGF reconstruction - probe {probe}", fontsize=11)
    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_residual_curves(results: pd.DataFrame, output_path: str, probe: str = "ols") -> str:
    """Residual 1 - R^2 versus K, one curve per (quench, family)."""
    subset = results[results["probe"] == probe]
    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    for (quench, family), group in subset.groupby(["quench", "family"]):
        curve = group.groupby("k")["r2"].mean()
        axis.plot(curve.index, 1.0 - curve.to_numpy(), marker="o", label=f"{family} - {quench}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("K")
    axis.set_ylabel("$1 - R^2$ (residual)")
    axis.set_title("Reconstruction residual, averaged over the time grid")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


CEILING_LABELS: tuple[str, ...] = (
    "walk",
    "+rrwp",
    "+laplacian",
    "+ring",
    "+heat/geo/res",
    "+interact/dist",
    "+classical dyn",
    "+two-body (Q)",
    "+mean-field",
)
#: 0-based index of the "classical structural ceiling" step.
CEILING_STEP: int = 5
AFM_COLOR = "#2a78d6"
FM_COLOR = "#eb6834"


def plot_ceiling_evolution(
    rf: pd.DataFrame,
    ols: pd.DataFrame,
    output_path: str,
    time_index: int = REFERENCE_TIME_INDEX,
) -> str:
    """Cumulative R^2 of the nine descriptor stacks (RF at the reference time, OLS
    time-averaged), colour = quench, line style = probe."""
    series: dict[str, dict[str, list[float]]] = {}
    for probe, frame, instant in (("rf", rf, True), ("ols", ols, False)):
        if instant:
            frame = frame[frame["time_index"] == time_index]
        pivot = frame.pivot_table(index="family", columns="quench", values="r2").reindex(
            list(CUMULATIVE_FAMILIES),
        )
        series[probe] = {"AFM": pivot["AFM"].tolist(), "FM": pivot["FM"].tolist()}

    steps = list(range(len(CEILING_LABELS)))
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.axvspan(-0.4, CEILING_STEP + 0.4, color="#f0efec", zorder=0, linewidth=0)
    styles = {
        ("rf", "AFM"): {"color": AFM_COLOR, "linestyle": "-", "marker": "o"},
        ("rf", "FM"): {"color": FM_COLOR, "linestyle": "-", "marker": "o"},
        ("ols", "AFM"): {"color": AFM_COLOR, "linestyle": (0, (4, 2)), "marker": "s"},
        ("ols", "FM"): {"color": FM_COLOR, "linestyle": (0, (4, 2)), "marker": "s"},
    }
    for (probe, quench), style in styles.items():
        axis.plot(
            steps,
            series[probe][quench],
            linewidth=2.0 if probe == "rf" else 1.6,
            markerfacecolor="white" if probe == "ols" else style["color"],
            markeredgecolor=style["color"],
            **style,
        )
    for quench, color in (("AFM", AFM_COLOR), ("FM", FM_COLOR)):
        final = series["rf"][quench][-1]
        axis.text(steps[-1] + 0.12, final, f"{quench} RF {final:.3f}", color=color, va="center")
    axis.set_xticks(steps)
    axis.set_xticklabels(CEILING_LABELS, rotation=30, ha="right", fontsize=8.5)
    axis.set_ylim(0.70, 0.95)
    axis.set_xlim(-0.5, len(CEILING_LABELS) + 0.8)
    axis.set_ylabel("Pooled reconstruction $R^2$")
    axis.set_title("Reconstructing quantum correlations from cumulative descriptor stacks")
    axis.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    legend = [
        Line2D([0], [0], color=AFM_COLOR, lw=2.4, label="AFM quench"),
        Line2D([0], [0], color=FM_COLOR, lw=2.4, label="FM quench"),
        Line2D([0], [0], color="#8a8984", marker="o", label="Random forest (graph folds)"),
        Line2D(
            [0],
            [0],
            color="#8a8984",
            linestyle=(0, (4, 2)),
            marker="s",
            markerfacecolor="white",
            label="OLS (in-sample, time-averaged)",
        ),
    ]
    axis.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5)
    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(os.path.splitext(output_path)[0] + ".png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
