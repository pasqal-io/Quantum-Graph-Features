"""Phase 1 bases: power values, RRWP, Laplacian augmentation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.design import column_scales, normalize_columns
from qgf.reconstruction.features_base import (
    canonical_eigenvectors,
    laplacian_basis,
    laplacian_features,
    normalized_laplacian,
    random_walk_matrix,
    rrwp_basis,
    walk_basis,
    walk_powers,
)

from .conftest import make_sample


def test_walk_powers_on_triangle():
    """Sur un triangle, (A^2)_ij = 1 et (A^3)_ij = 3 marches i->j de longueur 3."""
    adjacency = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pairs = np.triu_indices(3, k=1)
    powers = walk_powers(adjacency, 3, pairs)
    assert powers.shape == (3, 4)
    assert np.allclose(powers[:, 0], 0.0)  # (A^0)_ij = I_ij = 0 hors diagonale
    assert np.allclose(powers[:, 1], 1.0)
    assert np.allclose(powers[:, 2], 1.0)
    assert np.allclose(powers[:, 3], 3.0)


def test_walk_basis_first_column_is_identically_zero(tiny_samples, context):
    """Subtlety: the k=0 column does not provide an effective intercept."""
    block = walk_basis(tiny_samples[0], context)
    assert block.labels[0] == "walk_0"
    assert np.all(block.matrix[:, 0] == 0.0)


def test_walk_orders_allow_truncation(tiny_samples, context):
    block = walk_basis(tiny_samples[0], context)
    assert block.num_columns == context.k_max + 1
    for k in range(context.k_max + 1):
        assert block.select(k).num_columns == k + 1


def test_random_walk_matrix_is_row_stochastic():
    graph = nx.path_graph(4)
    matrix = random_walk_matrix(nx.to_numpy_array(graph))
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_random_walk_matrix_handles_isolated_node():
    """Zero-degree guard: no division by zero on an isolated node."""
    adjacency = np.zeros((3, 3))
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    matrix = random_walk_matrix(adjacency)
    assert np.all(np.isfinite(matrix))
    assert matrix[2].sum() == 0.0


def test_rrwp_values_on_path():
    """Sur 0-1-2, ((D^-1 A)^2)_02 = 1/2 : un pas de 0 vers 1 puis 1 vers 2."""
    sample = make_sample(nx.path_graph(3), 0)
    block = rrwp_basis(sample, BasisContext(k_max=2))
    rows, cols = sample.pair_index
    position = int(np.flatnonzero((rows == 0) & (cols == 2))[0])
    assert block.matrix[position, 2] == pytest.approx(0.5)


def test_normalized_laplacian_spectrum_in_zero_two(tiny_samples):
    """The bounded spectrum [0, 2] is the reason for choosing the normalised Laplacian."""
    for sample in tiny_samples:
        values = np.linalg.eigvalsh(normalized_laplacian(sample.adjacency))
        assert values.min() >= -1e-9
        assert values.max() <= 2.0 + 1e-9


def test_eigenvector_signs_are_canonical():
    """The sign is fixed by the largest-magnitude entry; ties go to the lowest index."""
    laplacian = normalized_laplacian(nx.to_numpy_array(nx.cycle_graph(6)))
    _, vectors = canonical_eigenvectors(laplacian)
    dominant = np.argmax(np.round(np.abs(vectors), 10), axis=0)
    signs = vectors[dominant, np.arange(vectors.shape[1])]
    assert np.all(signs > 0)


def test_laplacian_features_shape_and_padding():
    """3K columns, zero-padded when n < K."""
    sample = make_sample(nx.path_graph(4), 0)
    matrix = laplacian_features(sample.adjacency, 6, sample.pair_index)
    assert matrix.shape == (sample.num_pairs, 18)
    # Paires propres 5 et 6 inexistantes (n = 4) : colonnes nulles.
    assert np.all(matrix[:, 4:6] == 0.0)
    assert np.all(matrix[:, 10:12] == 0.0)
    assert np.all(matrix[:, 16:18] == 0.0)


def test_laplacian_basis_truncation_keeps_three_blocks(tiny_samples, context):
    block = laplacian_basis(tiny_samples[0], context)
    assert block.num_columns == 3 * context.k_max
    assert block.select(2).num_columns == 6
    assert set(block.select(2).labels) == {
        "lap_ui_1",
        "lap_ui_2",
        "lap_uj_1",
        "lap_uj_2",
        "lap_lambda_1",
        "lap_lambda_2",
    }


def test_normalize_columns_clamps_small_columns():
    """X / max(max|X|, 1) : columns of magnitude < 1 are not amplified."""
    matrix = np.array([[0.5, 4.0], [-0.25, -2.0]])
    scales = column_scales(matrix)
    assert scales.tolist() == [1.0, 4.0]
    normalized = normalize_columns(matrix)
    assert np.allclose(normalized[:, 0], matrix[:, 0])
    assert np.abs(normalized).max() <= 1.0
