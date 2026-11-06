"""Export of the reconstruction residual as a feature dataset (drop-in for benchmarks).

The pair residual r_ij(t) = C_ij(t) - OLS(classical ceiling) is fitted pooled over the
whole dataset and **all time steps** (one fit, static design), then scattered back into a
symmetric (T, N, N) tensor per graph. The one-body features receive symmetrically the node
residual onebody_i(t) - OLS(node ceiling), so node-level benchmarks also see a residual.

Everything stays in the released ZZ convention (``target_scale = 1``): the output is
directly comparable with the released ``*_residual`` datasets.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from qgf.reconstruction import registry
from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample, load_graph_samples, load_raw_dataset
from qgf.reconstruction.design import (
    build_pooled_design,
    normalize_columns,
    pooled_targets,
    with_constant,
)
from qgf.reconstruction.features_base import normalized_laplacian, random_walk_matrix
from qgf.reconstruction.features_optional import laplacian_pinv

LOGGER = logging.getLogger("qgf.reconstruction.residual_dataset")

#: k_max of the ceiling: the reference order of the extension protocol.
CEILING_K_MAX: int = 21
#: Cache cap: the ZINC ceiling designs exceed 512 MB but are reused across quenches.
CACHE_MAX_MB: float = 4096.0
#: Released name prefix per dataset.
RELEASED_PREFIX: dict[str, str] = {"MUTAG": "mutag", "ZINC_small": "zinc"}


def pooled_residual(ceiling_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pooled OLS residual over all time steps, via normal equations (static X).

    Args:
        ceiling_matrix: (P, C) normalised ceiling design, with intercept.
        targets: (P, N_t) pooled correlations.

    Returns:
        (P, N_t) residual r = Y - X beta, beta solving X^T X beta = X^T Y.
    """
    gram = ceiling_matrix.T @ ceiling_matrix
    moment = ceiling_matrix.T @ targets
    beta, *_ = np.linalg.lstsq(gram, moment, rcond=None)
    return targets - ceiling_matrix @ beta


def node_ceiling_design(sample: GraphSample, context: BasisContext) -> np.ndarray:
    """(N, C) node-level classical ceiling, the diagonal mirror of the pair ceiling.

    Purely structural / geometric node descriptors: degree, closed walks (A^k)_ii,
    random-walk returns ((D^-1 A)^k)_ii, heat-kernel diagonal exp(-tL)_ii, effective
    resistance L^+_ii, and local fields sum_j J_ij and sum_j (r_nn / d_ij).
    """
    adjacency = np.asarray(sample.adjacency, dtype=float)
    num_nodes = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    columns = [np.ones(num_nodes), degree]

    power = adjacency.copy()
    for _ in range(2, context.k_max + 1):
        power = power @ adjacency
        columns.append(np.diag(power).copy())
    walk = random_walk_matrix(adjacency)
    power_walk = walk.copy()
    for _ in range(2, context.k_max + 1):
        power_walk = power_walk @ walk
        columns.append(np.diag(power_walk).copy())

    values, vectors = np.linalg.eigh(normalized_laplacian(adjacency))
    for tau in context.heat_taus:
        heat = (vectors * np.exp(-tau * values)) @ vectors.T
        columns.append(np.diag(heat).copy())
    combinatorial = np.diag(degree) - adjacency
    columns.append(np.diag(laplacian_pinv(combinatorial)).copy())

    positions = sample.embedded_position
    if positions is not None:
        diff = positions[:, None, :] - positions[None, :, :]
        distance = np.sqrt((diff**2).sum(axis=-1))
        np.fill_diagonal(distance, np.inf)
        columns.append((context.c6 / distance**6).sum(axis=1))
        columns.append((context.r_nn_um / distance).sum(axis=1))
    return np.stack(columns, axis=1)


def node_residuals(
    samples: list[GraphSample],
    context: BasisContext,
) -> list[np.ndarray] | None:
    """One-body residual onebody - OLS(node ceiling), pooled over all nodes.

    Returns one (N, N_t) array per graph, or None when one-body features are missing.
    """
    if any(sample.occupations is None for sample in samples):
        return None
    designs = [node_ceiling_design(sample, context) for sample in samples]
    matrix = normalize_columns(np.concatenate(designs, axis=0))
    targets = np.concatenate([sample.occupations for sample in samples], axis=0)
    beta, *_ = np.linalg.lstsq(matrix.T @ matrix, matrix.T @ targets, rcond=None)
    residual = targets - matrix @ beta

    per_graph = []
    cursor = 0
    for sample in samples:
        width = sample.num_nodes
        per_graph.append(residual[cursor : cursor + width])
        cursor += width
    return per_graph


def scatter_to_tensor(
    residual_slice: np.ndarray,
    num_nodes: int,
    pair_index: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """(num_pairs, N_t) upper triangle -> symmetric (N_t, N, N) tensor, zero diagonal."""
    rows, cols = pair_index
    values = residual_slice.T  # (N_t, num_pairs)
    tensor = np.zeros((values.shape[0], num_nodes, num_nodes), dtype=np.float32)
    tensor[:, rows, cols] = values
    tensor[:, cols, rows] = values
    return tensor


def compute_residuals(
    dataset: str,
    quench: str,
    ceiling: str = registry.CLASSICAL_CEILING,
    cache_dir: str | None = None,
    max_graphs: int | None = None,
    root: str | None = None,
) -> tuple[list, dict]:
    """Residual ``Data`` objects (released layout, ZZ convention) and their meta."""
    samples = load_graph_samples(
        dataset,
        quench,
        target_scale=1.0,
        max_graphs=max_graphs,
        root=root,
    )
    context = BasisContext(
        k_max=CEILING_K_MAX,
        times_ns=tuple(samples[0].times_ns),
    )
    design = build_pooled_design(
        samples,
        ceiling,
        context,
        dataset,
        cache_dir,
        max_mb=CACHE_MAX_MB,
    )
    ceiling_matrix = normalize_columns(with_constant(design).matrix)
    targets = pooled_targets(samples)
    residual = pooled_residual(ceiling_matrix, targets)
    total = targets - targets.mean(axis=0, keepdims=True)
    LOGGER.info(
        "%s/%s: ceiling %d columns, unexplained variance (all times) %.4f",
        dataset,
        quench,
        ceiling_matrix.shape[1],
        float((residual**2).sum() / (total**2).sum()),
    )
    nodes = node_residuals(samples, context)

    raw, meta = load_raw_dataset(dataset, quench, root=root)
    if max_graphs is not None:
        raw = raw[:max_graphs]
    out = []
    cursor = 0
    for index, (sample, data) in enumerate(zip(samples, raw)):
        width = sample.num_pairs
        tensor = scatter_to_tensor(
            residual[cursor : cursor + width],
            sample.num_nodes,
            sample.pair_index,
        )
        cursor += width
        clone = data.clone()
        clone.twobody_features = torch.from_numpy(tensor)
        if nodes is not None:
            clone.onebody_features = torch.from_numpy(
                np.ascontiguousarray(nodes[index].T, dtype=np.float32),
            )
        out.append(clone)
    return out, meta


def emit_residual_dataset(
    dataset: str,
    quench: str,
    output_dir: str,
    ceiling: str = registry.CLASSICAL_CEILING,
    cache_dir: str | None = None,
    max_graphs: int | None = None,
    root: str | None = None,
) -> list[str]:
    """Builds and saves the residual dataset of one quench in the released format."""
    from qgf.data.io import save_dataset

    graphs, meta = compute_residuals(dataset, quench, ceiling, cache_dir, max_graphs, root)
    name = f"{RELEASED_PREFIX[dataset]}_{quench.lower()}_residual"
    base = {
        "name": name,
        "times_ns": meta["times_ns"],
        "zz_scale_applied": meta["zz_scale_applied"],  # residuals inherit the source scale
        "kind": "residual",
    }
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    if dataset == "ZINC_small" and max_graphs is None:
        offset = 0
        for split, size in zip(("train", "val", "test"), meta["split_sizes"]):
            path = Path(output_dir) / f"{name}_{split}.pt"
            save_dataset(path, graphs[offset : offset + size], {**base, "split": split})
            offset += size
            paths.append(str(path))
    else:
        path = Path(output_dir) / f"{name}.pt"
        save_dataset(path, graphs, {**base, "split": None})
        paths.append(str(path))
    LOGGER.info("residual dataset written: %s", paths)
    return paths


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG", choices=sorted(RELEASED_PREFIX))
    parser.add_argument("--quenches", nargs="+", default=["AFM", "FM"])
    parser.add_argument("--ceiling", default=registry.CLASSICAL_CEILING)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-dir", default="results/cache/reconstruction")
    parser.add_argument("--output-dir", default="results/reconstruction/residual_datasets")
    args = parser.parse_args(argv)
    for quench in args.quenches:
        emit_residual_dataset(
            args.dataset,
            quench,
            args.output_dir,
            ceiling=args.ceiling,
            cache_dir=args.cache_dir,
            max_graphs=args.max_graphs,
            root=args.data_root,
        )


if __name__ == "__main__":
    main()
