"""Very-short-time test, the pipeline diagnostic.

On 6-node graphs, exact diagonalisation is instantaneous; runs on the
MUTAG graphs go through Slurm.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgf.reconstruction.config import AFM_QUENCH, FM_QUENCH
from qgf.reconstruction.validation import (
    SHORT_TIMES_NS,
    check_short_time,
    couplings_for,
    short_time_passed,
    short_time_r2,
    simulate_short_time,
)

from .conftest import make_sample


@pytest.fixture
def small_samples():
    graphs = [nx.cycle_graph(6), nx.path_graph(6), nx.lollipop_graph(3, 2)]
    return [make_sample(graph, index) for index, graph in enumerate(graphs)]


def test_unweighted_couplings_follow_adjacency(small_samples):
    sample = small_samples[0]
    couplings = couplings_for(sample, "unweighted")
    assert np.allclose(couplings > 0, np.asarray(sample.adjacency) > 0)


def test_geometric_couplings_have_a_tail(small_samples):
    couplings = couplings_for(small_samples[0], "geometric")
    rows, cols = small_samples[0].pair_index
    non_bonded = np.asarray(small_samples[0].adjacency)[rows, cols] == 0
    assert np.all(couplings[rows, cols][non_bonded] > 0.0)


def test_simulation_replaces_correlations_and_times(small_samples):
    simulated = simulate_short_time(small_samples, AFM_QUENCH, times_ns=(10.0, 20.0))
    assert len(simulated) == len(small_samples)
    for original, short in zip(small_samples, simulated):
        assert short.correlations.shape == (2, original.num_nodes, original.num_nodes)
        assert short.times_ns.tolist() == [10.0, 20.0]
        assert np.abs(short.correlations).max() <= 0.25 + 1e-12


def test_simulation_skips_large_graphs(small_samples):
    with pytest.raises(ValueError, match="No graph"):
        simulate_short_time(small_samples, AFM_QUENCH, max_nodes=3)


def test_short_time_afm_saturates_on_walk_basis(small_samples):
    """Target exactly proportional to A_ij: any R² < 0.999 is a bug."""
    frame = short_time_r2(
        small_samples,
        AFM_QUENCH,
        families=("walk",),
        modes=("unweighted",),
    )
    assert frame["time_ns"].tolist() == list(SHORT_TIMES_NS)
    assert frame["r2"].min() > 0.999


def test_short_time_verdict_marks_mandatory_lines(small_samples):
    frame = short_time_r2(small_samples, AFM_QUENCH)
    verdict = check_short_time(frame)
    mandatory = verdict[verdict["mandatory"]]
    assert not mandatory.empty
    assert bool(mandatory["passed"].all())
    assert short_time_passed(verdict)


def test_geometric_coupling_is_not_in_the_walk_span(small_samples):
    """With the real J, the walk basis cannot saturate: the tail of J escapes it."""
    frame = short_time_r2(
        small_samples,
        AFM_QUENCH,
        families=("walk",),
        modes=("geometric",),
    )
    assert frame["r2"].max() < 0.999


def test_fm_short_time_law_is_not_universal(small_samples):
    """Delta = 2 delta_infini^G depends on the graph: the short-time law is no longer
    one."""
    frame = short_time_r2(
        small_samples,
        FM_QUENCH,
        families=("walk",),
        modes=("unweighted",),
    )
    verdict = check_short_time(frame)
    assert frame["r2"].max() < 0.999
    assert not verdict[verdict["quench"] == "FM"]["mandatory"].any()
