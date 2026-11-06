"""Optional Phase 3 bases: heat kernel, geodesic, resistance, chemistry."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.features_optional import (
    chemical_basis,
    geodesic_basis,
    heat_kernel_basis,
    resistance_basis,
)

from .conftest import make_sample


def test_heat_kernel_tends_to_identity_at_small_tau():
    """Tau -> 0 : exp(-tau L) -> I, so the off-diagonal entries vanish."""
    sample = make_sample(nx.cycle_graph(6), 0)
    context = BasisContext(k_max=2, heat_taus=(1e-6, 1.0))
    block = heat_kernel_basis(sample, context)
    assert block.num_columns == 2
    assert np.abs(block.matrix[:, 0]).max() < 1e-5
    assert np.abs(block.matrix[:, 1]).max() > 1e-3


def test_heat_kernel_decays_with_distance():
    sample = make_sample(nx.path_graph(5), 0)
    block = heat_kernel_basis(sample, BasisContext(k_max=2, heat_taus=(1.0,)))
    rows, cols = sample.pair_index
    adjacent = block.matrix[(cols - rows) == 1, 0]
    distant = block.matrix[(cols - rows) == 4, 0]
    assert adjacent.min() > distant.max()


def test_geodesic_matches_networkx():
    sample = make_sample(nx.path_graph(5), 0)
    context = BasisContext(k_max=2, max_geodesic=6)
    block = geodesic_basis(sample, context)
    rows, cols = sample.pair_index
    expected = np.array(
        [nx.shortest_path_length(sample.graph, int(i), int(j)) for i, j in zip(rows, cols)],
        dtype=float,
    )
    assert np.allclose(block.matrix[:, 0], expected)
    # One-hot consistent with the distance.
    for position, distance in enumerate(expected):
        assert block.matrix[position, int(distance)] == 1.0


def test_geodesic_marks_unreachable_pairs():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    sample = make_sample(graph, 0)
    context = BasisContext(k_max=2, max_geodesic=4)
    block = geodesic_basis(sample, context)
    labels = dict(zip(block.labels, range(block.num_columns)))
    rows, cols = sample.pair_index
    across = int(np.flatnonzero((rows == 0) & (cols == 2))[0])
    assert block.matrix[across, labels["geodesic_unreachable"]] == 1.0
    assert block.matrix[across, 0] == context.max_geodesic + 1


def test_resistance_on_path_is_series_sum():
    """On a path, resistances add in series: Omega_ij = |i - j|."""
    sample = make_sample(nx.path_graph(5), 0)
    block = resistance_basis(sample, BasisContext(k_max=2))
    rows, cols = sample.pair_index
    assert np.allclose(block.matrix[:, 0], (cols - rows).astype(float))
    assert np.allclose(block.matrix[:, 1], np.sqrt(cols - rows))


def test_resistance_on_cycle_is_parallel_combination():
    """On an N-cycle, Omega_ij = d (N - d) / N (two paths in parallel)."""
    sample = make_sample(nx.cycle_graph(6), 0)
    block = resistance_basis(sample, BasisContext(k_max=2))
    rows, cols = sample.pair_index
    distance = np.minimum(cols - rows, 6 - (cols - rows)).astype(float)
    assert np.allclose(block.matrix[:, 0], distance * (6 - distance) / 6)


def test_chemical_basis_encodes_bond_and_atom_types():
    sample = make_sample(nx.cycle_graph(4), 0, dataset="MUTAG")
    block = chemical_basis(sample, BasisContext(k_max=2))
    labels = dict(zip(block.labels, range(block.num_columns)))
    rows, cols = sample.pair_index
    bonded = np.asarray(sample.adjacency)[rows, cols] > 0
    assert np.all(block.matrix[bonded, labels["bond_is_0"]] == 1.0)
    assert np.all(block.matrix[~bonded, labels["bond_is_0"]] == 0.0)
    # All atoms are of type 0: each pair counts two.
    assert np.all(block.matrix[:, labels["atom_count_0"]] == 2.0)


def test_chemical_basis_requires_attributes():
    sample = make_sample(nx.cycle_graph(4), 0)
    sample.node_types = None
    sample.bond_types = None
    with pytest.raises(ValueError, match="no chemical attribute"):
        chemical_basis(sample, BasisContext(k_max=2))
