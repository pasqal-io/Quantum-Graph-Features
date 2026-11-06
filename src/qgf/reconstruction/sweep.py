"""Main sweep loop over (family, K, t, quench, probe).

Per-graph designs are independent of the time and of the quench (except the physical
controls): they are built once at ``k_max``, cached, then sliced by column order for every
K of the grid. Only the target changes. For ZINC, normal equations accumulated by blocks
avoid holding 3e6 x 51 in memory for every K.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from qgf.reconstruction import registry
from qgf.reconstruction.config import BasisContext, ProbeSpec, RunConfig
from qgf.reconstruction.data import GraphSample, load_graph_samples, validate_samples
from qgf.reconstruction.design import (
    DesignBlock,
    build_pooled_design,
    build_sample_design,
    column_scales,
    normalize_columns,
    pooled_groups,
    pooled_targets,
    resolve_family,
    with_constant,
)
from qgf.reconstruction.probes import (
    GramAccumulator,
    ProbeResult,
    kernel_ridge_cv,
    per_graph_r2,
    pooled_r2,
    random_forest_cv,
)

LOGGER = logging.getLogger(__name__)

#: Sanity test: expected total number of pairs per dataset.
EXPECTED_PAIRS: dict[str, int] = {"MUTAG": 30_505, "ZINC_small": 2_990_575}

#: Above this estimated pooled-design size, switch to normal equations.
STREAMING_THRESHOLD_BYTES: float = 2e9


def load_run_samples(config: RunConfig, quench: str) -> list[GraphSample]:
    """Samples of a run (time axis from the dataset metadata)."""
    samples = load_graph_samples(
        config.dataset,
        quench,
        None,
        target_scale=config.target_scale,
        max_graphs=config.max_graphs,
        root=config.data_root,
    )
    if not np.allclose(samples[0].times_ns, config.times()):
        raise ValueError(
            f"{config.dataset}/{quench}: dataset times {samples[0].times_ns} differ from the "
            f"configured grid {config.times()}",
        )
    return samples


def _phase_of(family: str) -> str:
    """Group of a composite family (the most advanced of its atoms)."""
    phases = {registry.available_bases()[atom] for atom in resolve_family(family)}
    return max(phases)


def _base_row(
    config: RunConfig,
    quench: str,
    family: str,
    probe: str,
    k: int,
    result: ProbeResult,
) -> dict:
    return {
        "dataset": config.dataset,
        "quench": quench,
        "family": family,
        "phase": _phase_of(family),
        "k": k,
        "probe": probe,
        "num_columns": result.num_columns,
        "num_parameters": result.num_parameters,
        "num_rows": result.num_rows,
        "fold_mode": "graph" if probe != "ols" else "in_sample",
        "intercept": config.intercept,
        "seed": config.seed,
    }


def _emit(
    rows: list[dict],
    config: RunConfig,
    quench: str,
    family: str,
    probe: str,
    k: int,
    result: ProbeResult,
    steps: Sequence[int],
    times_ns: np.ndarray,
) -> None:
    """Appends one row per evaluated time step."""
    adjusted = result.r2_adjusted
    fold_std = result.fold_std
    for position, step in enumerate(steps):
        row = _base_row(config, quench, family, probe, k, result)
        row.update(
            {
                "time_index": int(step),
                "time_ns": float(times_ns[step]),
                "r2": float(result.r2[position]),
                "r2_adjusted": float(adjusted[position]),
                "r2_fold_std": float(fold_std[position]),
            },
        )
        rows.append(row)


def _prepare(block: DesignBlock, config: RunConfig) -> DesignBlock:
    """Applies the explicit-intercept option."""
    return with_constant(block) if config.intercept else block


def _run_probe(
    probe_name: str,
    matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    config: RunConfig,
) -> ProbeResult:
    if probe_name == "ols":
        return pooled_r2(matrix, targets)
    if probe_name == "kernel_ridge":
        return kernel_ridge_cv(
            matrix,
            targets,
            groups,
            n_splits=config.n_splits,
            seed=config.seed,
        )
    if probe_name == "random_forest":
        return random_forest_cv(
            matrix,
            targets,
            groups,
            n_splits=config.n_splits,
            seed=config.seed,
        )
    raise ValueError(f"Unknown probe: {probe_name}")


def _probe_steps(probe: ProbeSpec, steps: Sequence[int]) -> tuple[int, ...]:
    """Time steps evaluated by a probe, intersected with those of the run."""
    if probe.time_indices is None:
        return tuple(steps)
    return tuple(step for step in probe.time_indices if step in set(steps))


def _dense_family_rows(
    config: RunConfig,
    quench: str,
    family: str,
    samples: Sequence[GraphSample],
    context: BasisContext,
    targets: np.ndarray,
    groups: np.ndarray,
    steps: Sequence[int],
) -> list[dict]:
    """Dense path: one pooled design in memory, sliced by K."""
    started = time.perf_counter()
    design = build_pooled_design(
        samples,
        family,
        context,
        dataset=config.dataset,
        cache_dir=config.cache_dir if config.cache_designs else None,
        max_mb=config.cache_max_mb,
    )
    LOGGER.info(
        "design %s/%s: %d rows x %d columns in %.1fs",
        quench,
        family,
        design.num_rows,
        design.num_columns,
        time.perf_counter() - started,
    )
    times_ns = samples[0].times_ns
    rows: list[dict] = []
    for probe in config.probes:
        probe_steps = _probe_steps(probe, steps)
        if not probe_steps:
            continue
        for k in probe.k_grid:
            block = _prepare(design.select(k), config)
            if block.num_columns == 0:
                continue
            if block.has_time_columns:
                # One column per time step: the design changes with the target.
                for step in probe_steps:
                    sub = block.select_time(step)
                    result = _run_probe(
                        probe.name,
                        normalize_columns(sub.matrix),
                        targets[:, [step]],
                        groups,
                        config,
                    )
                    _emit(rows, config, quench, family, probe.name, k, result, [step], times_ns)
            else:
                result = _run_probe(
                    probe.name,
                    normalize_columns(block.matrix),
                    targets[:, list(probe_steps)],
                    groups,
                    config,
                )
                _emit(rows, config, quench, family, probe.name, k, result, probe_steps, times_ns)
            LOGGER.info(
                "%s/%s/%s K=%d: %d columns, %.1fs cumulated",
                quench,
                family,
                probe.name,
                k,
                block.num_columns,
                time.perf_counter() - started,
            )
    return rows


def _streaming_family_rows(
    config: RunConfig,
    quench: str,
    family: str,
    samples: Sequence[GraphSample],
    context: BasisContext,
    steps: Sequence[int],
) -> list[dict]:
    """Streaming path: normal equations accumulated graph by graph.

    Two passes are needed: column normalisation is defined **after** pooling, so the
    scales must be known before accumulating the Gram matrix.
    """
    probes = [probe for probe in config.probes if probe.name == "ols"]
    if len(probes) != len(config.probes):
        raise ValueError("Streaming mode only supports the OLS probe")

    scales: np.ndarray | None = None
    reference: DesignBlock | None = None
    for sample in samples:
        block = _prepare(build_sample_design(sample, family, context), config)
        if block.has_time_columns:
            raise NotImplementedError(
                f"Family {family}: time-aligned columns are not supported in streaming mode",
            )
        local = column_scales(block.matrix)
        scales = local if scales is None else np.maximum(scales, local)
        reference = block
    assert scales is not None and reference is not None

    k_values = sorted({k for probe in probes for k in probe.k_grid})
    selections = {k: np.flatnonzero(reference.orders <= k) for k in k_values}
    accumulators = {
        k: GramAccumulator(selections[k].size, len(steps)) for k in k_values if selections[k].size
    }
    for sample in samples:
        block = _prepare(build_sample_design(sample, family, context), config)
        normalized = block.matrix / scales
        target = sample.targets()[:, list(steps)]
        for k, accumulator in accumulators.items():
            accumulator.add(normalized[:, selections[k]], target)

    times_ns = samples[0].times_ns
    rows: list[dict] = []
    for probe in probes:
        probe_steps = _probe_steps(probe, steps)
        positions = [list(steps).index(step) for step in probe_steps]
        for k in probe.k_grid:
            if k not in accumulators:
                continue
            full = accumulators[k].result()
            result = ProbeResult(
                r2=full.r2[positions],
                num_columns=full.num_columns,
                num_parameters=full.num_parameters,
                num_rows=full.num_rows,
            )
            _emit(rows, config, quench, family, probe.name, k, result, probe_steps, times_ns)
    return rows


def _check_k_grid(config: RunConfig) -> None:
    """Refuses a K larger than ``k_max``: the truncation would be silent."""
    requested = max((k for probe in config.probes for k in probe.k_grid), default=0)
    if requested > config.context.k_max:
        raise ValueError(
            f"K={requested} requested but k_max={config.context.k_max}: raise "
            "BasisContext.k_max to cover the whole grid",
        )


def _should_stream(config: RunConfig, samples: Sequence[GraphSample], family: str) -> bool:
    if config.streaming is not None:
        return config.streaming
    if any(probe.name != "ols" for probe in config.probes):
        return False
    total_pairs = sum(sample.num_pairs for sample in samples)
    columns = build_sample_design(samples[0], family, config.context).num_columns
    return total_pairs * columns * 8 > STREAMING_THRESHOLD_BYTES


def run_sweep(config: RunConfig) -> pd.DataFrame:
    """Runs the full sweep and returns the long-format result table."""
    _check_k_grid(config)
    frames: list[dict] = []
    for quench in config.quenches:
        samples = load_run_samples(config, quench)
        expected = EXPECTED_PAIRS.get(config.dataset) if config.max_graphs is None else None
        diagnostics = validate_samples(samples, config.dataset, quench, expected)
        LOGGER.info("%s", diagnostics.summary())
        context = config.context_for(quench)
        steps = (
            tuple(config.time_indices)
            if config.time_indices is not None
            else tuple(range(diagnostics.num_timesteps))
        )
        targets: np.ndarray | None = None
        groups: np.ndarray | None = None
        for family in config.families:
            if _should_stream(config, samples, family):
                LOGGER.info("family %s: streaming mode", family)
                frames.extend(
                    _streaming_family_rows(config, quench, family, samples, context, steps),
                )
                continue
            if targets is None:
                targets = pooled_targets(samples)
                groups = pooled_groups(samples)
            frames.extend(
                _dense_family_rows(
                    config,
                    quench,
                    family,
                    samples,
                    context,
                    targets,
                    groups,
                    steps,
                ),
            )
    return pd.DataFrame(frames)


def run_per_graph(config: RunConfig, family: str = "walk") -> pd.DataFrame:
    """Graph-by-graph OLS R^2 at K = n_alpha - 1 (Cayley-Hamilton test).

    At K = n_alpha - 1, Cayley-Hamilton makes higher powers redundant: the walk basis is
    saturated for that graph.
    """
    rows: list[dict] = []
    for quench in config.quenches:
        samples = load_run_samples(config, quench)
        context = config.context_for(quench)
        matrices = []
        targets = []
        for sample in samples:
            block = _prepare(
                build_sample_design(sample, family, context).select(sample.num_nodes - 1),
                config,
            )
            matrices.append(normalize_columns(block.matrix))
            targets.append(sample.targets())
        scores = per_graph_r2(matrices, targets)
        times_ns = samples[0].times_ns
        for index, sample in enumerate(samples):
            for step in range(scores.shape[1]):
                rows.append(
                    {
                        "dataset": config.dataset,
                        "quench": quench,
                        "family": family,
                        "graph_id": sample.graph_id,
                        "num_nodes": sample.num_nodes,
                        "k": sample.num_nodes - 1,
                        "time_index": step,
                        "time_ns": float(times_ns[step]),
                        "r2": float(scores[index, step]),
                    },
                )
    return pd.DataFrame(rows)
