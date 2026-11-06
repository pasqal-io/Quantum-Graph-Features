"""Entry point of the reconstruction-ceiling and residual-analysis paper items.

Subcommands (run from the repository root):

    ceiling            extended basis, nine cumulative stacks, OLS + random forest (MUTAG),
                       summary table and figure (unexplained variance with the full basis)
    residual-analysis  spectrum, range, rank and dTWA tests of the ZINC residual
    retention          residual retention of predictive power, from benchmark results
    check-residuals    compare regenerated residual datasets with the released ones

Heavy sweeps also run through ``qgf.reconstruction.run`` (see README.md). Every
subcommand accepts ``--output-dir`` (default ``results/reconstruction/...``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from qgf.reconstruction import analysis
from qgf.reconstruction import run as recon_run
from qgf.reconstruction.report import (
    ceiling_summary,
    plot_ceiling_evolution,
    read_results,
)

PULSE_COLOR = {"AFM": "#2a78d6", "FM": "#eb6834"}
SOURCE_STYLE = {
    "initial": {"linestyle": "-", "linewidth": 2.2},
    "reconstruction": {"linestyle": (0, (5, 2)), "linewidth": 2.0},
    "residual": {"linestyle": (0, (1, 1.6)), "linewidth": 2.2, "marker": "o", "markersize": 4.5},
}
CAT_COLORS = {"Edge": "#0072B2", "Node": "#E69F00", "Chemical": "#CC79A7", "Graph": "#009E73"}
MUTED = "#8a8984"


# --------------------------------------------------------------------------------------
# ceiling
# --------------------------------------------------------------------------------------


def command_ceiling(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    common = ["--dataset", args.dataset, "--stacks", "cumulative", "--no-figures"]
    common += ["--cache-dir", args.cache_dir]
    if args.max_graphs:
        common += ["--max-graphs", str(args.max_graphs)]
    ols_dir, rf_dir = out / "ols", out / "rf"
    if not args.skip_sweeps:
        recon_run.main(
            ["sweep", *common, "--probes", "ols", "--k-grid", "21", "--times", "all"]
            + ["--intercept", "--output-dir", str(ols_dir)],
        )
        recon_run.main(
            ["sweep", *common, "--probes", "random_forest", "--nonlinear-k", "21"]
            + ["--nonlinear-times", "reference", "--output-dir", str(rf_dir)],
        )
    ols = read_results(str(ols_dir / "results.parquet"))
    rf = read_results(str(rf_dir / "results.parquet"))
    summary = ceiling_summary(pd.concat([ols, rf], ignore_index=True))
    summary.to_csv(out / "ceiling_summary.csv", index=False)
    plot_ceiling_evolution(rf, ols, str(out / "ceiling_evolution.pdf"))
    final = summary[summary["step"] == summary["step"].max()]
    print(final[["probe", "quench", "r2_ref", "r2_time_avg", "unexplained_ref"]].to_string())


# --------------------------------------------------------------------------------------
# residual-analysis
# --------------------------------------------------------------------------------------


def _plot_nature(profiles: dict, retention_medians: dict | None, path: Path) -> None:
    mosaic = [["A", "B"], ["C", "C"]] if retention_medians else [["B", "C"]]
    figure, axes = plt.subplot_mosaic(mosaic, figsize=(13.0, 10.5 if retention_medians else 5.2))
    if retention_medians:
        ax = axes["A"]
        order = sorted(retention_medians, key=retention_medians.get)
        ax.barh(
            range(len(order)),
            [retention_medians[c] * 100 for c in order],
            color=[CAT_COLORS[c] for c in order],
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlim(0, 105)
        ax.set_xlabel("mean retention vs full correlation (%)")
        ax.set_title("A - What the residual predicts, by target locality", loc="left")
    ax = axes["B"]
    for quench, prof in profiles.items():
        dmax = int(np.max(np.flatnonzero(prof["dist_count"] > 0)))
        d = np.arange(1, dmax + 1)
        for source in analysis.SOURCES:
            ax.plot(
                d,
                prof["dist_mean"][source][1 : dmax + 1],
                color=PULSE_COLOR[quench],
                **SOURCE_STYLE[source],
            )
    ax.set_yscale("log")
    ax.set_xlabel("shortest-path distance d(i, j)")
    ax.set_ylabel("mean |value| (ZZ convention)")
    ax.set_title("B - Range: reconstruction short-range, residual long-range", loc="left")
    ax = axes["C"]
    for quench, prof in profiles.items():
        for source in analysis.SOURCES:
            ax.plot(
                prof["lam_centers"],
                prof["energy_cdf"][source],
                color=PULSE_COLOR[quench],
                **SOURCE_STYLE[source],
            )
    ax.plot([0, 2], [0, 1], color=MUTED, lw=1.0, ls=(0, (2, 3)))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("normalised Laplacian eigenvalue (low = global / long-wavelength)")
    ax.set_ylabel("cumulative energy")
    ax.set_title("C - Spectral organisation", loc="left")
    handles = [Line2D([], [], color=c, lw=2.4, label=f"{q} quench") for q, c in PULSE_COLOR.items()]
    handles += [Line2D([], [], color=MUTED, label=s, **SOURCE_STYLE[s]) for s in analysis.SOURCES]
    axes["C"].legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def command_residual_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles, ranks, classical = {}, [], []
    for quench in ("AFM", "FM"):
        samples, residuals = analysis.load_pairs(quench, stride=1, root=args.data_root)
        if args.max_graphs:
            samples, residuals = samples[: args.max_graphs], residuals[: args.max_graphs]
        sub = slice(None, None, args.stride)
        profiles[quench] = analysis.spectral_range_profiles(
            samples[sub],
            residuals[sub],
            centroid_over_time=args.stride == 1,
        )
        rank = analysis.rank_test(samples, residuals)
        ranks.append({"quench": quench, **{k: v for k, v in rank.items() if np.isscalar(v)}})
        if not args.skip_classical:
            cls_sub = slice(None, None, args.classical_stride)
            test = analysis.classical_test(
                samples[cls_sub],
                residuals[cls_sub],
                quench,
            )
            classical.append(
                {"quench": quench, **{k: v for k, v in test.items() if np.isscalar(v)}},
            )
    tables = analysis.profiles_table(profiles)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(ranks).to_csv(out / "rank_test.csv", index=False)
    if classical:
        pd.DataFrame(classical).to_csv(out / "classical_test.csv", index=False)
    medians = None
    if args.retention_table and os.path.exists(args.retention_table):
        summary = analysis.retention_summary(pd.read_csv(args.retention_table))
        medians = dict(zip(summary["category"], summary["mean"]))
    _plot_nature(profiles, medians, out / "residual_nature.pdf")
    print(tables["centroid"].to_string(index=False))
    print(pd.DataFrame(ranks).to_string(index=False))
    if classical:
        print(pd.DataFrame(classical).to_string(index=False))


# --------------------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------------------


def command_retention(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = analysis.retention(
        analysis.load_benchmark_results(args.residual_results),
        analysis.load_benchmark_results(args.full_results),
        residual_pe=args.residual_pe,
        full_pe=args.full_pe,
        random_from=args.random_from,
    )
    table.to_csv(out / "retention.csv", index=False)
    summary = analysis.retention_summary(table)
    summary.to_csv(out / "retention_summary.csv", index=False)
    print(table.to_string(index=False))
    print(summary.to_string(index=False))


# --------------------------------------------------------------------------------------
# check-residuals
# --------------------------------------------------------------------------------------


def compare_residual_files(generated: Path, released: Path) -> dict:
    """Max absolute and relative deviation of twobody / onebody residual features."""
    from qgf.data.io import load_dataset

    mine, meta_mine = load_dataset(generated)
    ref, meta_ref = load_dataset(released)
    if len(mine) != len(ref):
        raise ValueError(f"{generated.name}: {len(mine)} graphs vs {len(ref)} released")
    stats = {"file": generated.name, "num_graphs": len(mine)}
    for field in ("twobody_features", "onebody_features"):
        delta = max(float((a[field] - b[field]).abs().max()) for a, b in zip(mine, ref))
        scale = max(float(b[field].abs().max()) for b in ref)
        rms = np.sqrt(
            sum(float(((a[field] - b[field]) ** 2).sum()) for a, b in zip(mine, ref))
            / sum(float((b[field] ** 2).sum()) for b in ref),
        )
        stats[f"{field}_max_abs_delta"] = delta
        stats[f"{field}_max_abs_released"] = scale
        stats[f"{field}_relative_rms_delta"] = float(rms)
    stats["times_match"] = meta_mine["times_ns"] == meta_ref["times_ns"]
    return stats


def command_check_residuals(args: argparse.Namespace) -> None:
    from qgf.data import DEFAULT_ROOT

    released_root = Path(args.data_root) if args.data_root else DEFAULT_ROOT
    rows = []
    for path in sorted(Path(args.generated).glob("*_residual*.pt")):
        rows.append(compare_residual_files(path, released_root / path.name))
    frame = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "residual_dataset_check.csv", index=False)
    with pd.option_context("display.width", 250):
        print(frame.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ceiling = sub.add_parser("ceiling")
    ceiling.add_argument("--dataset", default="MUTAG")
    ceiling.add_argument("--max-graphs", type=int, default=None)
    ceiling.add_argument("--skip-sweeps", action="store_true", help="only rebuild summary/figure")
    ceiling.add_argument("--cache-dir", default="results/cache/reconstruction")
    ceiling.add_argument("--output-dir", default="results/reconstruction/ceiling")
    ceiling.set_defaults(handler=command_ceiling)

    resid = sub.add_parser("residual-analysis")
    resid.add_argument("--stride", type=int, default=5, help="graph subsampling of profiles")
    resid.add_argument("--classical-stride", type=int, default=5)
    resid.add_argument("--skip-classical", action="store_true")
    resid.add_argument("--max-graphs", type=int, default=None)
    resid.add_argument(
        "--retention-table", default="results/reconstruction/retention/retention.csv"
    )
    resid.add_argument("--data-root", default=None)
    resid.add_argument("--output-dir", default="results/reconstruction/residual_analysis")
    resid.set_defaults(handler=command_residual_analysis)

    ret = sub.add_parser("retention")
    ret.add_argument("--residual-results", required=True)
    ret.add_argument("--full-results", required=True)
    ret.add_argument("--residual-pe", default="FM_AFM")
    ret.add_argument("--full-pe", default="FM_AFM")
    ret.add_argument("--random-from", default="full", choices=["full", "residual"])
    ret.add_argument("--output-dir", default="results/reconstruction/retention")
    ret.set_defaults(handler=command_retention)

    check = sub.add_parser("check-residuals")
    check.add_argument("--generated", default="results/reconstruction/residual_datasets")
    check.add_argument("--data-root", default=None)
    check.add_argument("--output-dir", default="results/reconstruction/residual_datasets")
    check.set_defaults(handler=command_check_residuals)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
