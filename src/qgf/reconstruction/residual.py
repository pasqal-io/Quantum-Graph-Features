"""Characterisation of the reconstruction residual by beyond-WL descriptors.

Question: is what escapes the classical ceiling describable by pure graph theory? The
residual

    r_p = y_p - y_hat_p        with y_hat = pooled OLS(ceiling) at the reference time

is reconstructed with a lean beyond-WL stack (treewidth >= 3 motifs, automorphism
orbits). A high R^2(residual | beyond-WL) would mean graph structure beyond the ~2-WL
expressivity of the ceiling; R^2 ~ 0 means it is geometric or dynamical.

Controls:
  - *within-graph*: residual demeaned per graph (removes the per-graph offset that the
    single pooled coefficient vector cannot fit);
  - *ceiling(control)*: reconstructing the residual with the ceiling itself; the OLS
    being in-sample, orthogonality forces R^2 ~ 0 (consistency check).
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from qgf.reconstruction import registry
from qgf.reconstruction.config import REFERENCE_TIME_INDEX, BasisContext, RunConfig
from qgf.reconstruction.data import validate_samples
from qgf.reconstruction.design import (
    build_pooled_design,
    normalize_columns,
    pooled_groups,
    pooled_targets,
    with_constant,
)
from qgf.reconstruction.probes import permutation_group_importance, pooled_r2, random_forest_cv
from qgf.reconstruction.residual_dataset import CEILING_K_MAX
from qgf.reconstruction.sweep import EXPECTED_PAIRS, load_run_samples

LOGGER = logging.getLogger("qgf.reconstruction.residual")


def group_demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtracts the per-graph mean."""
    sums = np.bincount(groups, weights=values)
    counts = np.bincount(groups)
    means = sums / np.maximum(counts, 1)
    return values - means[groups]


def ceiling_residual(ceiling_matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """In-sample OLS residual of the ceiling reconstruction (with intercept)."""
    beta, *_ = np.linalg.lstsq(ceiling_matrix, target, rcond=None)
    return target - ceiling_matrix @ beta


#: Columns carried by each beyond-WL atom.
_ATOM_LABELS: dict[str, tuple[str, ...]] = {
    "motif": ("co_clique4", "co_open4"),
    "symmetry": ("same_orbit", "orbit_size_min", "orbit_size_max"),
}


def _column_groups(labels: tuple[str, ...]) -> dict[str, list[int]]:
    """Column indices of every family atom, for grouped importance."""
    label_to_atom = {
        label: atom for atom, atom_labels in _ATOM_LABELS.items() for label in atom_labels
    }
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        atom = label_to_atom.get(label)
        if atom is not None:
            groups.setdefault(atom, []).append(index)
    return groups


def _reconstruction_rows(
    dataset: str,
    quench: str,
    label: str,
    reconstructor: str,
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> list[dict]:
    """In-sample OLS and held-out RF (graph folds) reconstructing ``target``."""
    ols = float(pooled_r2(matrix, target).r2[0])
    forest = float(
        random_forest_cv(matrix, target[:, None], groups, mode="graph", seed=seed).r2[0],
    )
    return [
        {
            "dataset": dataset,
            "quench": quench,
            "target": label,
            "reconstructor": reconstructor,
            "probe": probe,
            "r2": value,
            "num_columns": int(matrix.shape[1]),
            "num_rows": int(matrix.shape[0]),
        }
        for probe, value in (("ols", ols), ("rf", forest))
    ]


def run_residual(
    dataset: str,
    quenches: tuple[str, ...],
    ceiling: str,
    beyond: str,
    output_dir: str,
    cache_dir: str | None,
    max_graphs: int | None,
    seed: int,
    time_index: int = REFERENCE_TIME_INDEX,
    data_root: str | None = None,
) -> pd.DataFrame:
    """Computes and reconstructs the residual for every quench; returns the long table."""
    template = RunConfig(dataset=dataset, max_graphs=max_graphs, data_root=data_root)
    context = BasisContext(k_max=CEILING_K_MAX, times_ns=tuple(float(t) for t in template.times()))

    rows: list[dict] = []
    importances: list[dict] = []
    for quench in quenches:
        samples = load_run_samples(template, quench)
        expected = EXPECTED_PAIRS.get(dataset) if max_graphs is None else None
        LOGGER.info("%s", validate_samples(samples, dataset, quench, expected).summary())

        ceiling_design = build_pooled_design(samples, ceiling, context, dataset, cache_dir)
        ceiling_matrix = normalize_columns(with_constant(ceiling_design).matrix)
        beyond_design = build_pooled_design(samples, beyond, context, dataset, cache_dir)
        beyond_matrix = normalize_columns(beyond_design.matrix)
        groups = pooled_groups(samples)

        target = pooled_targets(samples)[:, time_index]
        residual = ceiling_residual(ceiling_matrix, target)
        within = group_demean(residual, groups)
        LOGGER.info(
            "%s/%s: ceiling %d columns, residual var=%.3e (within %.3e), beyond %d columns",
            dataset,
            quench,
            ceiling_matrix.shape[1],
            float(residual.var()),
            float(within.var()),
            beyond_matrix.shape[1],
        )

        plan = [
            ("residual_raw", "beyond", beyond_matrix, residual),
            ("residual_within", "beyond", beyond_matrix, within),
            ("residual_raw", "ceiling", ceiling_matrix, residual),
            ("target_full", "beyond", beyond_matrix, target),
        ]
        for label, reconstructor, matrix, values in plan:
            rows.extend(
                _reconstruction_rows(
                    dataset, quench, label, reconstructor, matrix, values, groups, seed
                ),
            )

        attribution = permutation_group_importance(
            beyond_matrix,
            within[:, None],
            groups,
            _column_groups(beyond_design.labels),
            mode="graph",
            seed=seed,
        )
        for name, drop in attribution.items():
            importances.append(
                {
                    "dataset": dataset,
                    "quench": quench,
                    "target": "residual_within",
                    "group": name,
                    "delta_r2": float(np.asarray(drop)[0]),
                },
            )

    results = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, "residual_reconstruction.csv"), index=False)
    pd.DataFrame(importances).to_csv(
        os.path.join(output_dir, "residual_importance.csv"),
        index=False,
    )
    LOGGER.info("Results written to %s", output_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--quenches", nargs="+", default=["AFM", "FM"])
    parser.add_argument("--ceiling", default=registry.CLASSICAL_CEILING)
    parser.add_argument("--beyond", default=registry.BEYOND_WL)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-dir", default="results/cache/reconstruction")
    parser.add_argument("--output-dir", default="results/reconstruction/residual_beyond_wl")
    args = parser.parse_args(argv)
    results = run_residual(
        dataset=args.dataset,
        quenches=tuple(args.quenches),
        ceiling=args.ceiling,
        beyond=args.beyond,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_graphs=args.max_graphs,
        seed=args.seed,
        data_root=args.data_root,
    )
    view = results[results["reconstructor"] == "beyond"]
    LOGGER.info(
        "Residual reconstruction (beyond-WL):\n%s",
        view.pivot_table(index=["quench", "target"], columns="probe", values="r2").to_string(),
    )


if __name__ == "__main__":
    main()
