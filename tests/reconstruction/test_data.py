"""Input contract: pair indexing, conversion of released Data objects, validations."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from qgf.reconstruction.data import (
    GraphSample,
    pair_index,
    sample_from_data,
    validate_samples,
)

from .conftest import TIMES_NS, make_sample


def test_pair_count_matches_strict_upper_triangle(tiny_samples):
    """P = sum_alpha N_alpha (N_alpha - 1) / 2."""
    expected = sum(sample.num_nodes * (sample.num_nodes - 1) // 2 for sample in tiny_samples)
    assert sum(sample.num_pairs for sample in tiny_samples) == expected
    for sample in tiny_samples:
        rows, cols = sample.pair_index
        assert rows.size == sample.num_pairs
        assert np.all(rows < cols)


def test_targets_follow_pair_index_order(tiny_samples):
    """The target follows the lexicographic order of the strict upper triangle."""
    sample = tiny_samples[0]
    rows, cols = sample.pair_index
    targets = sample.targets()
    assert targets.shape == (sample.num_pairs, sample.correlations.shape[0])
    for position in (0, 3, sample.num_pairs - 1):
        for step in range(sample.correlations.shape[0]):
            assert targets[position, step] == pytest.approx(
                sample.correlations[step, rows[position], cols[position]],
            )


def test_pair_index_helper_matches_numpy():
    rows, cols = pair_index(5)
    expected_rows, expected_cols = np.triu_indices(5, k=1)
    assert np.array_equal(rows, expected_rows)
    assert np.array_equal(cols, expected_cols)


def _released_data(num_nodes: int = 4, num_steps: int = 3) -> Data:
    graph = nx.cycle_graph(num_nodes)
    edges = np.array(list(graph.edges())).T
    edge_index = np.concatenate([edges, edges[::-1]], axis=1)
    rng = np.random.default_rng(0)
    twobody = rng.normal(size=(num_steps, num_nodes, num_nodes))
    twobody = twobody + np.swapaxes(twobody, 1, 2)
    for step in range(num_steps):
        np.fill_diagonal(twobody[step], 0.0)
    attr = np.zeros((edge_index.shape[1], 4))
    attr[:, 0] = 1.0
    return Data(
        x=torch.eye(num_nodes),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(attr, dtype=torch.float32),
        pos=torch.tensor(rng.normal(size=(num_nodes, 2)), dtype=torch.float32),
        twobody_features=torch.tensor(twobody, dtype=torch.float32),
        onebody_features=torch.tensor(rng.uniform(size=(num_steps, num_nodes))),
    )


def test_sample_from_released_data_scales_and_transposes():
    """ZZ correlations are scaled by target_scale; one-body becomes node-major."""
    data = _released_data()
    sample = sample_from_data(data, 3, np.array([1.0, 2.0, 3.0]), target_scale=0.25)
    assert sample.graph_id == 3
    assert np.allclose(sample.correlations, 0.25 * data.twobody_features.numpy(), atol=1e-7)
    assert sample.occupations.shape == (4, 3)
    assert np.allclose(sample.occupations, data.onebody_features.numpy().T)
    assert sample.embedded_position.shape == (4, 2)
    assert np.array_equal(sample.adjacency, nx.to_numpy_array(nx.cycle_graph(4)))
    assert sample.bond_types[0, 1] == 0 and sample.bond_types[0, 2] == -1


def test_sample_from_data_rejects_wrong_time_axis():
    with pytest.raises(ValueError, match="twobody_features of shape"):
        sample_from_data(_released_data(), 0, np.arange(5.0))


def test_validate_samples_reports_diagnostics(tiny_samples):
    diagnostics = validate_samples(tiny_samples, "TEST", "AFM")
    assert diagnostics.num_graphs == len(tiny_samples)
    assert diagnostics.num_pairs == sum(sample.num_pairs for sample in tiny_samples)
    assert diagnostics.num_timesteps == 4
    assert diagnostics.times_last_ns == TIMES_NS[-1]
    assert "P=" in diagnostics.summary()


def test_validate_samples_rejects_wrong_pair_count(tiny_samples):
    with pytest.raises(ValueError, match="sanity"):
        validate_samples(tiny_samples, "TEST", "AFM", expected_pairs=1)


def test_validate_samples_rejects_asymmetric_adjacency(tiny_samples):
    sample = tiny_samples[0]
    broken = GraphSample(
        graph_id=0,
        adjacency=np.triu(sample.adjacency),
        correlations=sample.correlations,
        times_ns=sample.times_ns,
    )
    with pytest.raises(ValueError, match="non-symmetric adjacency"):
        validate_samples([broken], "TEST", "AFM")


def test_validate_samples_rejects_asymmetric_correlations():
    sample = make_sample(nx.cycle_graph(4), 0)
    sample.correlations[0, 0, 1] += 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        validate_samples([sample], "TEST", "AFM")


def test_aromatic_index_depends_on_dataset(tiny_graphs):
    """MUTAG encodes aromaticity (index 0), ZINC does not."""
    assert make_sample(tiny_graphs[0], 0, dataset="MUTAG").aromatic_bond_index == 0
    assert make_sample(tiny_graphs[0], 0, dataset="ZINC_small").aromatic_bond_index is None
