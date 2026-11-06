"""Very-short-time test, the discriminating unit test of the pipeline.

At the first non-trivial order in t, the connected correlation is proportional to the
pair coupling: any R^2 < 0.999 at t <= 30 ns signals an indexing or normalisation bug,
not a physical effect.

The stored grid starts at 100 ns, where Omega t is already ~1.6 rad: the perturbative
regime is not reached. The test data are therefore **simulated** exactly on graphs small
enough for dense diagonalisation.

Two coupling variants are reported:

* ``unweighted``: J_ij = J_NN A_ij. The target is exactly proportional to A_ij, so this
  is the variant that must pass 0.999 on the walk basis;
* ``geometric``: J_ij = C6/d_ij^6, the actual register physics. The target is then
  proportional to J_ij, which the walk basis cannot represent exactly (diagnostic).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from qgf.reconstruction.config import C6_RAD_UM6_PER_US, R_NN_UM, BasisContext, QuenchSpec
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import build_pooled_design, normalize_columns, pooled_targets
from qgf.reconstruction.probes import pooled_r2
from qgf.reconstruction.quench import (
    MAX_EXACT_NODES,
    interaction_matrix,
    nearest_neighbour_coupling,
    quench_correlations,
)

LOGGER = logging.getLogger(__name__)

#: Default very short times, in ns.
SHORT_TIMES_NS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)


def couplings_for(
    sample: GraphSample,
    mode: str,
    c6: float = C6_RAD_UM6_PER_US,
    r_nn: float = R_NN_UM,
) -> np.ndarray:
    """Coupling matrix (rad/us) of the test variant."""
    if mode == "unweighted":
        return nearest_neighbour_coupling(r_nn=r_nn, c6=c6) * np.asarray(
            sample.adjacency,
            dtype=float,
        )
    if mode == "geometric":
        if sample.embedded_position is None:
            raise ValueError(f"Graph {sample.graph_id}: no positions")
        return interaction_matrix(sample.embedded_position, c6=c6)
    raise ValueError(f"Unknown coupling mode: {mode}")


def simulate_short_time(
    samples: Sequence[GraphSample],
    quench: QuenchSpec,
    times_ns: Sequence[float] = SHORT_TIMES_NS,
    mode: str = "unweighted",
    max_nodes: int = MAX_EXACT_NODES,
    c6: float = C6_RAD_UM6_PER_US,
    r_nn: float = R_NN_UM,
) -> list[GraphSample]:
    """Replaces the correlations by an exact simulation at very short times.

    Only graphs with N <= ``max_nodes`` are kept (dense diagonalisation in 2^N).
    """
    times = np.asarray(times_ns, dtype=float)
    times_us = times / 1000.0
    j_nn = nearest_neighbour_coupling(r_nn=r_nn, c6=c6)
    simulated: list[GraphSample] = []
    for sample in samples:
        if sample.num_nodes > max_nodes:
            continue
        couplings = couplings_for(sample, mode, c6=c6, r_nn=r_nn)
        correlations = quench_correlations(
            couplings,
            times_us,
            omega=quench.omega(j_nn),
            delta=quench.delta(couplings),
        )
        simulated.append(replace(sample, correlations=correlations, times_ns=times))
    if not simulated:
        raise ValueError(f"No graph with N <= {max_nodes} in the sample")
    return simulated


def short_time_r2(
    samples: Sequence[GraphSample],
    quench: QuenchSpec,
    families: Sequence[str] = ("walk", "jpow"),
    modes: Sequence[str] = ("unweighted", "geometric"),
    times_ns: Sequence[float] = SHORT_TIMES_NS,
    k: int = 1,
    max_nodes: int = MAX_EXACT_NODES,
    context: BasisContext | None = None,
) -> pd.DataFrame:
    """Pooled R^2 at very short times, per (coupling mode, family, t)."""
    base_context = context or BasisContext(k_max=max(k, 1), quench=quench)
    rows = []
    for mode in modes:
        simulated = simulate_short_time(
            samples,
            quench,
            times_ns=times_ns,
            mode=mode,
            max_nodes=max_nodes,
        )
        targets = pooled_targets(simulated)
        for family in families:
            design = build_pooled_design(
                simulated,
                family,
                base_context,
                dataset="short_time",
                cache_dir=None,
            )
            block = design.select(k)
            result = pooled_r2(normalize_columns(block.matrix), targets)
            for step, value in enumerate(result.r2):
                rows.append(
                    {
                        "quench": quench.name,
                        "coupling": mode,
                        "family": family,
                        "k": k,
                        "time_ns": float(times_ns[step]),
                        "r2": float(value),
                        "num_graphs": len(simulated),
                        "num_rows": result.num_rows,
                    },
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ShortTimeExpectation:
    """(quench, coupling, family) combination and expected threshold."""

    quench: str
    coupling: str
    family: str
    threshold: float = 0.999
    mandatory: bool = True
    comment: str = ""


#: Only AFM + unweighted coupling + walk basis truly saturates; it is the pipeline test.
#: In FM, delta = 2 delta_inf^G depends on the graph, so the short-time proportionality
#: constant varies between graphs; with geometric couplings the target is a nonlinear
#: function of J_ij, outside the span of matrix powers of J~.
SHORT_TIME_EXPECTATIONS: tuple[ShortTimeExpectation, ...] = (
    ShortTimeExpectation(
        quench="AFM",
        coupling="unweighted",
        family="walk",
        threshold=0.999,
        mandatory=True,
        comment="pipeline test: target exactly proportional to A_ij",
    ),
    ShortTimeExpectation(
        quench="AFM",
        coupling="geometric",
        family="jpow",
        threshold=0.98,
        mandatory=False,
        comment="diagnostic: linear J~ misses the nonlinearity in J",
    ),
    ShortTimeExpectation(
        quench="FM",
        coupling="unweighted",
        family="walk",
        threshold=0.90,
        mandatory=False,
        comment="diagnostic: delta_inf^G breaks the universality of the short-time law",
    ),
)


def check_short_time(
    frame: pd.DataFrame,
    expectations: Sequence[ShortTimeExpectation] = SHORT_TIME_EXPECTATIONS,
) -> pd.DataFrame:
    """Verdict of the test, one row per expected (quench, coupling, family, t)."""
    rows = []
    for expectation in expectations:
        selected = frame[
            (frame["quench"] == expectation.quench)
            & (frame["coupling"] == expectation.coupling)
            & (frame["family"] == expectation.family)
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "quench": expectation.quench,
                    "coupling": expectation.coupling,
                    "family": expectation.family,
                    "time_ns": row["time_ns"],
                    "r2": row["r2"],
                    "threshold": expectation.threshold,
                    "mandatory": expectation.mandatory,
                    "passed": row["r2"] >= expectation.threshold,
                    "comment": expectation.comment,
                },
            )
    return pd.DataFrame(rows)


def short_time_passed(verdict: pd.DataFrame) -> bool:
    """True if every mandatory row of the verdict passes."""
    mandatory = verdict[verdict["mandatory"]]
    return bool(mandatory["passed"].all()) and not mandatory.empty
