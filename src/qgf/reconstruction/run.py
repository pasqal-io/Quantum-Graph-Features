"""CLI of the reconstruction-ceiling experiment.

Examples:

    # Base families, OLS over the whole (K, t) grid, MUTAG, both quenches
    python -m qgf.reconstruction.run sweep --dataset MUTAG \
        --families walk rrwp walk+laplacian rrwp+laplacian --probes ols

    # Nonlinear probes at the reference time index
    python -m qgf.reconstruction.run sweep --dataset MUTAG --probes kernel_ridge \
        random_forest --nonlinear-k 3 7 21 --nonlinear-times reference

    # Extended basis (cumulative stacks), random forest, graph folds
    python -m qgf.reconstruction.run sweep --dataset MUTAG --stacks cumulative \
        --probes random_forest --nonlinear-k 21
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from qgf.reconstruction import registry
from qgf.reconstruction.config import (
    K_GRID_NONLINEAR,
    K_GRID_OLS,
    REFERENCE_TIME_INDEX,
    BasisContext,
    ProbeSpec,
    RunConfig,
)
from qgf.reconstruction.data import validate_samples
from qgf.reconstruction.reference import coherence_report, gap_report, per_graph_summary
from qgf.reconstruction.report import (
    ceiling_summary,
    extension_report,
    plot_heatmaps,
    plot_residual_curves,
    quench_contrast,
    read_results,
    table_at_time,
    table_time_averaged,
    to_latex,
    write_results,
)
from qgf.reconstruction.sweep import EXPECTED_PAIRS, load_run_samples, run_per_graph, run_sweep
from qgf.reconstruction.validation import (
    SHORT_TIMES_NS,
    check_short_time,
    short_time_passed,
    short_time_r2,
)

LOGGER = logging.getLogger("qgf.reconstruction")

STACKS: dict[str, tuple[str, ...]] = {
    "cumulative": registry.CUMULATIVE_FAMILIES,
    "baselines": registry.BASELINE_FAMILIES,
}


def _serialize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return {key: _serialize(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def dump_config(config: RunConfig, output_dir: str) -> str:
    """Dumps the run configuration (one config file per run)."""
    payload = {key: _serialize(value) for key, value in dataclasses.asdict(config).items()}
    payload["times_ns"] = config.times().tolist()
    payload["created_at"] = datetime.now(UTC).isoformat()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _resolve_time_indices(argument: str | None) -> tuple[int, ...] | None:
    """--times: "all", "reference" or a list of indices."""
    if argument in (None, "all"):
        return None
    if argument == "reference":
        return (REFERENCE_TIME_INDEX,)
    return tuple(int(item) for item in argument.split(","))


def build_config(args: argparse.Namespace) -> RunConfig:
    """RunConfig from the command-line arguments."""
    time_indices = _resolve_time_indices(args.times)
    probes = []
    for name in args.probes:
        k_grid = tuple(args.k_grid) if name == "ols" else tuple(args.nonlinear_k)
        probe_times = time_indices if name == "ols" else _resolve_time_indices(args.nonlinear_times)
        probes.append(ProbeSpec(name=name, k_grid=k_grid, time_indices=probe_times))

    template = RunConfig(dataset=args.dataset)
    # k_max covers every requested grid, otherwise select(k) would silently truncate.
    context = BasisContext(
        k_max=max(k for probe in probes for k in probe.k_grid),
        max_ring_size=args.max_ring_size,
        times_ns=tuple(float(value) for value in template.times()),
    )
    families = STACKS[args.stacks] if args.stacks else tuple(args.families)
    return dataclasses.replace(
        template,
        context=context,
        quenches=tuple(args.quenches),
        families=tuple(families),
        probes=tuple(probes),
        time_indices=time_indices,
        intercept=args.intercept,
        n_splits=args.n_splits,
        seed=args.seed,
        max_graphs=args.max_graphs,
        streaming=args.streaming,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cache_designs=not args.no_cache,
        data_root=args.data_root,
    )


def _write_tables(results: pd.DataFrame, output_dir: str) -> None:
    """Regenerates Tables 13/14/15 (CSV and LaTeX) and the cumulative-stack summary."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    table13 = table_time_averaged(results, probe="ols")
    if not table13.empty:
        table13.to_csv(os.path.join(tables_dir, "table13_time_averaged.csv"))
        with open(os.path.join(tables_dir, "table13_time_averaged.tex"), "w") as handle:
            handle.write(
                to_latex(table13, "Pooled OLS R^2 averaged over time.", "tab:recon-t13"),
            )
    for index, quench in enumerate(("AFM", "FM"), start=14):
        table = table_at_time(results, quench=quench)
        if table.empty:
            continue
        name = f"table{index}_{quench.lower()}_tindex{REFERENCE_TIME_INDEX}"
        table.to_csv(os.path.join(tables_dir, f"{name}.csv"))
        with open(os.path.join(tables_dir, f"{name}.tex"), "w") as handle:
            handle.write(
                to_latex(
                    table,
                    f"R^2 per probe and family, {quench}, time index {REFERENCE_TIME_INDEX} "
                    f"({table.attrs['time_ns']:.0f} ns).",
                    f"tab:recon-t{index}",
                ),
            )
    summary = ceiling_summary(results)
    if not summary.empty:
        summary.to_csv(os.path.join(tables_dir, "ceiling_summary.csv"), index=False)


def command_sweep(args: argparse.Namespace) -> None:
    """Sweep (family, K, t, quench, probe) and its deliverables."""
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)
    dump_config(config, config.output_dir)
    results = run_sweep(config)
    path = write_results(results, os.path.join(config.output_dir, "results.parquet"))
    LOGGER.info("%d result rows written to %s", len(results), path)
    _write_tables(results, config.output_dir)

    gaps = gap_report(results)
    gaps.to_csv(os.path.join(config.output_dir, "gap_report.csv"), index=False)
    for _, row in gaps.dropna(subset=["observed"]).iterrows():
        LOGGER.info(
            "  %-24s expected %.3f observed %.3f gap %+.3f",
            row["label"],
            row["expected"],
            row["observed"],
            row["gap"],
        )
    extensions = extension_report(results)
    if not extensions.empty:
        extensions.to_csv(os.path.join(config.output_dir, "extension_report.csv"), index=False)
    contrast = quench_contrast(results)
    if not contrast.empty:
        contrast.to_csv(os.path.join(config.output_dir, "quench_contrast.csv"), index=False)
    if not args.no_figures:
        figures_dir = os.path.join(config.output_dir, "figures")
        for probe in results["probe"].unique():
            plot_heatmaps(results, os.path.join(figures_dir, f"heatmap_{probe}.pdf"), probe=probe)
        if "ols" in set(results["probe"]):
            plot_residual_curves(results, os.path.join(figures_dir, "residual_vs_k.pdf"))


def command_per_graph(args: argparse.Namespace) -> None:
    """Graph-by-graph OLS at K = n_alpha - 1 (Cayley-Hamilton)."""
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)
    frames = [run_per_graph(config, family=family) for family in config.families]
    per_graph = pd.concat(frames, ignore_index=True)
    write_results(per_graph, os.path.join(config.output_dir, "per_graph.parquet"))
    summary = per_graph_summary(per_graph)
    summary.to_csv(os.path.join(config.output_dir, "per_graph_summary.csv"), index=False)
    LOGGER.info("Per-graph summary:\n%s", summary.to_string(index=False))
    if args.pooled_results and os.path.exists(args.pooled_results):
        coherence = coherence_report(read_results(args.pooled_results), per_graph)
        coherence.to_csv(
            os.path.join(config.output_dir, "coherence_pool_vs_per_graph.csv"),
            index=False,
        )


def command_short_time(args: argparse.Namespace) -> None:
    """t <= 30 ns test on exactly simulated data."""
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)
    times = (
        tuple(float(item) for item in args.short_times.split(","))
        if args.short_times
        else SHORT_TIMES_NS
    )
    frames = []
    for quench in config.quenches:
        samples = load_run_samples(config, quench)
        LOGGER.info("%s", validate_samples(samples, config.dataset, quench).summary())
        frames.append(
            short_time_r2(
                samples,
                config.context_for(quench).quench,
                times_ns=times,
                max_nodes=args.max_nodes,
            ),
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(os.path.join(config.output_dir, "short_time.csv"), index=False)
    verdict = check_short_time(frame)
    verdict.to_csv(os.path.join(config.output_dir, "short_time_verdict.csv"), index=False)
    LOGGER.info("Short-time verdict:\n%s", verdict.to_string(index=False))
    if not short_time_passed(verdict):
        LOGGER.error("Short-time test FAILED: indexing or normalisation suspect")


def command_report(args: argparse.Namespace) -> None:
    """Regenerates tables and gap report from a result file."""
    results = read_results(args.results)
    os.makedirs(args.output_dir, exist_ok=True)
    _write_tables(results, args.output_dir)
    gap_report(results).to_csv(os.path.join(args.output_dir, "gap_report.csv"), index=False)


def command_bases(args: argparse.Namespace) -> None:
    """Inventory of the registry."""
    for name, phase in sorted(registry.available_bases().items(), key=lambda kv: (kv[1], kv[0])):
        print(f"{phase}  {name}")
    print(f"\nExpected pairs per dataset: {EXPECTED_PAIRS}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="MUTAG", choices=["MUTAG", "ZINC_small"])
    parser.add_argument("--quenches", nargs="+", default=["AFM", "FM"])
    parser.add_argument("--families", nargs="+", default=["walk", "rrwp"])
    parser.add_argument("--stacks", choices=sorted(STACKS), default=None)
    parser.add_argument("--probes", nargs="+", default=["ols"])
    parser.add_argument("--k-grid", nargs="+", type=int, default=list(K_GRID_OLS))
    parser.add_argument("--nonlinear-k", nargs="+", type=int, default=list(K_GRID_NONLINEAR))
    parser.add_argument("--times", default="all", help="all | reference | i,j,k")
    parser.add_argument("--nonlinear-times", default="reference", help="all | reference | i,j,k")
    parser.add_argument("--intercept", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--max-ring-size", type=int, default=8)
    parser.add_argument("--streaming", dest="streaming", action="store_const", const=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_const", const=False)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-dir", default="results/cache/reconstruction")
    parser.add_argument("--output-dir", default="results/reconstruction/run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser("sweep", help="sweep (family, K, t, quench, probe)")
    _add_common(sweep)
    sweep.set_defaults(handler=command_sweep)

    per_graph = subparsers.add_parser("per-graph", help="per-graph OLS at K = n-1")
    _add_common(per_graph)
    per_graph.add_argument("--pooled-results", default=None)
    per_graph.set_defaults(handler=command_per_graph)

    short_time = subparsers.add_parser("short-time", help="simulated t <= 30 ns test")
    _add_common(short_time)
    short_time.add_argument("--short-times", default=None, help="times in ns, comma-separated")
    short_time.add_argument("--max-nodes", type=int, default=12)
    short_time.set_defaults(handler=command_short_time)

    report = subparsers.add_parser("report", help="regenerate tables")
    report.add_argument("--results", required=True)
    report.add_argument("--output-dir", default="results/reconstruction/report")
    report.set_defaults(handler=command_report)

    bases = subparsers.add_parser("bases", help="registry inventory")
    bases.set_defaults(handler=command_bases)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
