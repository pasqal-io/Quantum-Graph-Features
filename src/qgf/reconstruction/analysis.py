"""Analysis of the reconstruction residual on ZINC (spectrum, range, rank, dTWA, retention).

Three tensors per graph, all in the released ZZ convention: C (full correlation,
``zinc_{q}``), R (residual, ``zinc_{q}_residual``) and the reconstruction C - R.

* **Spectrum**: each tensor is projected on the normalised-Laplacian eigenbasis,
  M = U^T X U, energy E_kl = M_kl^2 binned by the mean eigenvalue (lambda_k + lambda_l)/2.
  Low lambda = global, long-wavelength modes.
* **Range**: mean |value| versus shortest-path distance d(i, j).
* **Rank**: is the residual a rank-1 uniform (sum-rule) component? (top-mode energy,
  overlap with the uniform vector, tail versus N, tail after per-graph demeaning).
* **Classical dynamics**: R^2 of the long-range residual on the dTWA correlation of the
  same quench.
* **Retention**: residual predictive power normalised between random features (0) and
  the full correlations (1), per target locality, from intrinsic-benchmark results.

All analyses read the reference time index (7) of the stored grid.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

from qgf.reconstruction.config import (
    QUENCHES,
    REFERENCE_TIME_INDEX,
    BasisContext,
)
from qgf.reconstruction.data import GraphSample, load_graph_samples, load_raw_dataset
from qgf.reconstruction.features_base import normalized_laplacian

SOURCES: tuple[str, ...] = ("initial", "reconstruction", "residual")
NUM_LAMBDA_BINS: int = 24
MAX_DISTANCE: int = 10
LONG_RANGE_MIN: int = 4


def load_pairs(
    quench: str,
    dataset: str = "ZINC_small",
    stride: int = 1,
    root: str | None = None,
) -> tuple[list[GraphSample], list[np.ndarray]]:
    """Full-correlation samples (ZZ) and residual tensors (T, N, N), subsampled by stride."""
    samples = load_graph_samples(dataset, quench, target_scale=1.0, root=root)
    residual, _ = load_raw_dataset(dataset, quench, suffix="_residual", root=root)
    if len(residual) != len(samples):
        raise ValueError(f"{len(residual)} residual graphs for {len(samples)} graphs")
    picked = range(0, len(samples), stride)
    return (
        [samples[i] for i in picked],
        [np.asarray(residual[i].twobody_features, dtype=float) for i in picked],
    )


def _distances(sample: GraphSample) -> np.ndarray:
    """(n_pairs,) shortest-path distance of every pair (MAX_DISTANCE + 1 if beyond)."""
    rows, cols = sample.pair_index
    lengths = dict(nx.all_pairs_shortest_path_length(sample.graph))
    return np.array(
        [lengths.get(int(i), {}).get(int(j), MAX_DISTANCE + 1) for i, j in zip(rows, cols)],
    )


def spectral_range_profiles(
    samples: list[GraphSample],
    residuals: list[np.ndarray],
    time_index: int = REFERENCE_TIME_INDEX,
    min_nodes: int = 4,
    centroid_over_time: bool = False,
) -> dict:
    """Energy CDF over lambda, spectral centroid and range profile of the three sources."""
    lam_edges = np.linspace(0.0, 2.0, NUM_LAMBDA_BINS + 1)
    num_steps = samples[0].correlations.shape[0]
    energy = {s: np.zeros(NUM_LAMBDA_BINS) for s in SOURCES}
    centroid = {s: np.zeros(num_steps) for s in SOURCES}
    dist_abs = {s: np.zeros(MAX_DISTANCE + 1) for s in SOURCES}
    dist_count = np.zeros(MAX_DISTANCE + 1)
    graphs = 0
    for sample, residual in zip(samples, residuals):
        if sample.num_nodes < min_nodes:
            continue
        full = sample.correlations
        tensors = {"initial": full, "residual": residual, "reconstruction": full - residual}
        values, vectors = np.linalg.eigh(normalized_laplacian(sample.adjacency))
        level = 0.5 * (values[:, None] + values[None, :])
        bins = np.clip(np.digitize(level.ravel(), lam_edges) - 1, 0, NUM_LAMBDA_BINS - 1)
        rows, cols = sample.pair_index
        distance = _distances(sample)
        keep = distance <= MAX_DISTANCE
        np.add.at(dist_count, distance[keep], 1)
        for name, tensor in tensors.items():
            if centroid_over_time:
                power = np.einsum("ki,tij,jl->tkl", vectors.T, tensor, vectors) ** 2
                total = power.sum(axis=(1, 2))
                total = np.where(total > 0, total, 1.0)
                centroid[name] += (power * level[None]).sum(axis=(1, 2)) / total
                at_ref = power[time_index]
            else:
                at_ref = (vectors.T @ tensor[time_index] @ vectors) ** 2
                total = at_ref.sum()
                centroid[name][time_index] += (at_ref * level).sum() / (total if total > 0 else 1)
            energy[name] += np.bincount(bins, weights=at_ref.ravel(), minlength=NUM_LAMBDA_BINS)
            np.add.at(dist_abs[name], distance[keep], np.abs(tensor[time_index][rows, cols][keep]))
        graphs += 1
    safe = np.where(dist_count > 0, dist_count, 1.0)
    return {
        "lam_centers": 0.5 * (lam_edges[:-1] + lam_edges[1:]),
        "energy_cdf": {s: np.cumsum(e) / max(e.sum(), 1e-300) for s, e in energy.items()},
        "centroid": {s: c / max(graphs, 1) for s, c in centroid.items()},
        "dist_mean": {s: d / safe for s, d in dist_abs.items()},
        "dist_count": dist_count,
        "num_graphs": graphs,
    }


def profiles_table(profiles: dict[str, dict], time_index: int = REFERENCE_TIME_INDEX) -> dict:
    """Long tables (energy CDF, range profile, centroid) of per-quench profiles."""
    energy, ranges, centroids = [], [], []
    for quench, prof in profiles.items():
        for source in SOURCES:
            for lam, cdf in zip(prof["lam_centers"], prof["energy_cdf"][source]):
                energy.append({"quench": quench, "source": source, "lambda": lam, "cdf": cdf})
            for d in range(1, MAX_DISTANCE + 1):
                if prof["dist_count"][d] > 0:
                    ranges.append(
                        {
                            "quench": quench,
                            "source": source,
                            "distance": d,
                            "mean_abs": prof["dist_mean"][source][d],
                            "num_pairs": int(prof["dist_count"][d]),
                        },
                    )
            centroids.append(
                {
                    "quench": quench,
                    "source": source,
                    "centroid_ref": prof["centroid"][source][time_index],
                },
            )
    return {
        "energy_cdf": pd.DataFrame(energy),
        "range": pd.DataFrame(ranges),
        "centroid": pd.DataFrame(centroids),
    }


# --------------------------------------------------------------------------------------
# Rank-1 / sum-rule test
# --------------------------------------------------------------------------------------


def rank_test(
    samples: list[GraphSample],
    residuals: list[np.ndarray],
    time_index: int = REFERENCE_TIME_INDEX,
) -> dict:
    """Is the long-range residual a rank-1 uniform (sum-rule) component?"""
    dist_raw = np.zeros(MAX_DISTANCE + 1)
    dist_dem = np.zeros(MAX_DISTANCE + 1)
    dist_count = np.zeros(MAX_DISTANCE + 1)
    top1, overlap, tail = [], [], []
    for sample, tensor in zip(samples, residuals):
        n = sample.num_nodes
        if n < 4:
            continue
        resid = tensor[time_index]
        rows, cols = sample.pair_index
        pair_mean = resid[rows, cols].mean()
        values, vectors = np.linalg.eigh(resid)
        energy = values**2
        top = int(np.argmax(energy))
        top1.append(energy[top] / energy.sum() if energy.sum() > 0 else 0.0)
        overlap.append(float(np.dot(vectors[:, top], np.ones(n) / np.sqrt(n)) ** 2))
        distance = _distances(sample)
        keep = distance <= MAX_DISTANCE
        values_pairs = resid[rows, cols]
        np.add.at(dist_count, distance[keep], 1)
        np.add.at(dist_raw, distance[keep], np.abs(values_pairs[keep]))
        np.add.at(dist_dem, distance[keep], np.abs(values_pairs[keep] - pair_mean))
        far = keep & (distance >= LONG_RANGE_MIN)
        if far.any():
            tail.append((n, float(np.abs(values_pairs[far]).mean())))
    safe = np.where(dist_count > 0, dist_count, 1.0)
    tail_arr = np.array(tail)
    ns, vals = tail_arr[:, 0], tail_arr[:, 1]
    mask = (ns > 0) & (vals > 0)
    raw, dem = dist_raw / safe, dist_dem / safe
    return {
        "top1_energy_median": float(np.median(top1)),
        "uniform_overlap_median": float(np.median(overlap)),
        "tail_slope_vs_n": float(np.polyfit(np.log(ns[mask]), np.log(vals[mask]), 1)[0]),
        "tail_kept_after_demean": float(
            dem[LONG_RANGE_MIN:].sum() / max(raw[LONG_RANGE_MIN:].sum(), 1e-300),
        ),
        "dist_raw": raw,
        "dist_demeaned": dem,
        "top1_energy": np.array(top1),
        "uniform_overlap": np.array(overlap),
        "tail_by_n": tail_arr,
    }


# --------------------------------------------------------------------------------------
# Classical dynamics (dTWA) test
# --------------------------------------------------------------------------------------


def _r2_single(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of a one-feature regression = squared Pearson correlation."""
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def classical_test(
    samples: list[GraphSample],
    residuals: list[np.ndarray],
    quench: str,
    time_index: int = REFERENCE_TIME_INDEX,
) -> dict:
    """R^2 of the residual on the dTWA correlation of the same quench, by range."""
    from qgf.reconstruction.features_extension import classical_dynamics_basis

    context = replace(
        BasisContext(k_max=1),
        quench=QUENCHES[quench],
        times_ns=tuple(float(t) for t in samples[0].times_ns),
    )
    pooled = defaultdict(list)
    for sample, tensor in zip(samples, residuals):
        if sample.num_nodes < 4:
            continue
        rows, cols = sample.pair_index
        distance = _distances(sample)
        keep = distance <= MAX_DISTANCE
        classical = classical_dynamics_basis(sample, context).matrix[:, time_index]
        pooled["resid"].append(tensor[time_index][rows, cols][keep])
        pooled["quantum"].append(sample.correlations[time_index][rows, cols][keep])
        pooled["classical"].append(4.0 * classical[keep])  # occupation -> ZZ
        pooled["d"].append(distance[keep])
    data = {key: np.concatenate(value) for key, value in pooled.items()}
    far = data["d"] >= LONG_RANGE_MIN
    near = ~far
    return {
        "r2_resid_long": _r2_single(data["classical"][far], data["resid"][far]),
        "r2_full_long": _r2_single(data["classical"][far], data["quantum"][far]),
        "r2_resid_short": _r2_single(data["classical"][near], data["resid"][near]),
        "num_long_range_pairs": int(far.sum()),
        "pooled": data,
    }


# --------------------------------------------------------------------------------------
# Retention of predictive power (intrinsic benchmark results)
# --------------------------------------------------------------------------------------

CATEGORIES: dict[str, tuple[str, ...]] = {
    "Node": (
        "degree_prediction",
        "betweenness_prediction",
        "bridge_node_prediction",
        "node_eccentricity",
        "smallest_cycle_node",
    ),
    "Edge": (
        "link_prediction",
        "edge_betweenness",
        "smallest_cycle_edge",
        "is_bridge_edge",
        "local_edge_connectivity",
    ),
    "Graph": (
        "num_cycles",
        "graph_diameter",
        "graph_density",
        "wiener_index",
        "algebraic_connectivity",
    ),
    "Chemical": ("logp", "tpsa", "fraction_csp3", "num_rotatable_bonds", "bond_type"),
}


def load_benchmark_results(root: str) -> pd.DataFrame:
    """Benchmark scores from ``<root>/<pe_type>/<benchmark>/result.json`` files or a CSV.

    Returns columns (pe_type, benchmark, metric, value), averaged over repeated entries.
    A CSV must already hold these four columns.
    """
    if root.endswith(".csv"):
        frame = pd.read_csv(root)
    else:
        records = []
        for path in glob.glob(os.path.join(root, "*", "*", "result.json")):
            with open(path) as handle:
                record = json.load(handle)
            records.append(
                {
                    "pe_type": record["pe_type"],
                    "benchmark": record["benchmark"],
                    "metric": record["metric"],
                    "value": float(record["raw_value"]),
                },
            )
        frame = pd.DataFrame(records)
    return frame.groupby(["pe_type", "benchmark", "metric"], as_index=False)["value"].mean()


def retention(
    residual_runs: pd.DataFrame,
    full_runs: pd.DataFrame,
    residual_pe: str = "FM_AFM",
    full_pe: str = "FM_AFM",
    random_pe: str = "random",
    random_from: str = "full",
) -> pd.DataFrame:
    """Per-benchmark retention (residual - random) / (full - random), sign-aware.

    ``residual_runs`` holds the residual dataset scores, ``full_runs`` the full-matrix
    dataset scores. The random-feature baseline is read from ``random_from`` ("full" or
    "residual"); both runs carry one, which differ only by training noise.
    """
    random_runs = full_runs if random_from == "full" else residual_runs
    index = ["benchmark"]
    pick = {
        name: frame[frame["pe_type"] == pe].set_index(index)
        for name, frame, pe in (
            ("random", random_runs, random_pe),
            ("residual", residual_runs, residual_pe),
            ("full", full_runs, full_pe),
        )
    }
    rows = []
    for category, benches in CATEGORIES.items():
        for bench in benches:
            if not all(bench in frame.index for frame in pick.values()):
                continue
            metric = pick["residual"].loc[bench, "metric"]
            rand, res, full = (float(pick[n].loc[bench, "value"]) for n in pick)
            lower_better = metric == "mae"
            den = (rand - full) if lower_better else (full - rand)
            num = (rand - res) if lower_better else (res - rand)
            rows.append(
                {
                    "category": category,
                    "benchmark": bench,
                    "metric": metric,
                    "random": rand,
                    "residual": res,
                    "full": full,
                    "retention": num / den if abs(den) > 1e-12 else np.nan,
                },
            )
    return pd.DataFrame(rows)


def retention_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and median retention per category (the paper quotes the mean, in %)."""
    return (
        table.groupby("category")["retention"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )
