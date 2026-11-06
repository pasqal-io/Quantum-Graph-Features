"""Phase 2 bases: geometry, J interaction, cycles, classical Ising."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgf.reconstruction.config import AFM_QUENCH, FM_QUENCH, BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.features_extension import (
    distance_basis,
    geometry_basis,
    interaction_basis,
    interaction_laplacian_basis,
    interaction_power_basis,
    interaction_rrwp_basis,
    mean_field_basis,
    radial_basis,
    scaled_interaction,
    susceptibility_betas,
    two_body_basis,
    two_body_curve_basis,
)
from qgf.reconstruction.features_ring import ring_features

from .conftest import TIMES_NS, make_sample


def _pair_position(sample: GraphSample, first: int, second: int) -> int:
    rows, cols = sample.pair_index
    return int(np.flatnonzero((rows == first) & (cols == second))[0])


def test_geometry_columns_are_in_rnn_units(tiny_samples, context):
    """Distances are made dimensionless by r_NN, so the shortest bond equals
    1."""
    sample = tiny_samples[0]
    block = geometry_basis(sample, context)
    labels = dict(zip(block.labels, range(block.num_columns)))
    assert block.matrix[:, labels["geo_d1"]].min() == pytest.approx(1.0)
    assert block.matrix[:, labels["geo_inv6"]].max() == pytest.approx(1.0)
    assert block.matrix[:, labels["geo_d2"]].min() == pytest.approx(1.0)


def test_geometry_has_expected_column_count(tiny_samples, context):
    block = geometry_basis(tiny_samples[0], context)
    # 2 puissances directes + 5 inverses + 16 RBF + 4 statistiques de voisinage.
    assert block.num_columns == 2 + 5 + context.rbf_centers + 4
    assert not block.has_time_columns


def test_geometry_requires_positions(context):
    sample = make_sample(nx.cycle_graph(4), 0)
    sample.embedded_position = None
    with pytest.raises(ValueError, match="no atom positions"):
        geometry_basis(sample, context)


def test_radial_basis_peaks_at_centre():
    centers = np.array([1.0, 2.0])
    values = radial_basis(np.array([1.0]), centers, width=0.5)
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] < 0.2


def test_geometry_is_finite_with_coincident_atoms(context):
    """Coincident positions: inverse powers are set to zero."""
    sample = make_sample(nx.path_graph(3), 0)
    sample.embedded_position = np.zeros((3, 2))
    block = geometry_basis(sample, context)
    assert np.all(np.isfinite(block.matrix))


def test_scaled_interaction_is_one_at_nearest_neighbour(tiny_samples, context):
    couplings = scaled_interaction(tiny_samples[0], context)
    assert np.allclose(np.diag(couplings), 0.0)
    assert couplings.max() == pytest.approx(1.0)


def test_interaction_tail_is_nonzero_off_edges(tiny_samples, context):
    """Unlike A, J is never strictly zero off the edges."""
    sample = tiny_samples[0]
    couplings = scaled_interaction(sample, context)
    rows, cols = sample.pair_index
    non_bonded = np.asarray(sample.adjacency)[rows, cols] == 0
    values = couplings[rows, cols][non_bonded]
    assert np.all(values > 0.0)
    assert np.all(values < 1.0)


def test_interaction_power_basis_starts_at_order_one(tiny_samples, context):
    block = interaction_power_basis(tiny_samples[0], context)
    assert block.labels[0] == "jpow_1"
    assert block.num_columns == context.k_max
    assert block.select(1).num_columns == 1


def test_interaction_rrwp_rows_are_stochastic(tiny_samples, context):
    block = interaction_rrwp_basis(tiny_samples[0], context)
    assert block.num_columns == context.k_max
    assert np.all(np.isfinite(block.matrix))


def test_interaction_laplacian_has_three_blocks(tiny_samples, context):
    block = interaction_laplacian_basis(tiny_samples[0], context)
    assert block.num_columns == 3 * context.k_max
    assert block.labels[0] == "jlap_ui_1"


def test_susceptibility_betas_stay_in_paramagnetic_regime(tiny_samples, context):
    couplings = scaled_interaction(tiny_samples[0], context)
    radius = float(np.abs(np.linalg.eigvalsh(couplings)).max())
    betas = susceptibility_betas(couplings, context)
    assert betas.size == context.beta_count
    assert np.all(betas * radius < 1.0)
    assert np.all(np.diff(betas) > 0)


def test_mean_field_resums_interaction_powers(tiny_samples, context):
    """chi = (1 - beta J~)^-1 must equal the series sum_k beta^k J~^k."""
    sample = tiny_samples[0]
    couplings = scaled_interaction(sample, context)
    block = mean_field_basis(sample, context)
    betas = susceptibility_betas(couplings, context)
    rows, cols = sample.pair_index
    beta = betas[0]
    series = np.zeros_like(couplings)
    power = np.eye(couplings.shape[0])
    for order in range(1, 40):
        power = power @ couplings
        series += beta**order * power
    assert np.allclose(block.matrix[:, 0], series[rows, cols], atol=1e-8)


def test_two_body_basis_is_time_aligned(tiny_samples, context):
    """One column per time step, only one active at a time."""
    block = two_body_basis(tiny_samples[0], context)
    assert block.has_time_columns
    assert block.num_columns == len(context.times_ns)
    assert block.select_time(2).num_columns == 1
    assert block.select_time(2).labels == ("twobody_t2",)


def test_two_body_curve_keeps_all_columns_static(tiny_samples, context):
    block = two_body_curve_basis(tiny_samples[0], context)
    assert not block.has_time_columns
    assert block.select_time(0).num_columns == len(context.times_ns)


def test_two_body_values_depend_on_the_quench(tiny_samples):
    afm = BasisContext(k_max=2, quench=AFM_QUENCH, times_ns=tuple(TIMES_NS))
    fm = BasisContext(k_max=2, quench=FM_QUENCH, times_ns=tuple(TIMES_NS))
    values_afm = two_body_basis(tiny_samples[0], afm).matrix
    values_fm = two_body_basis(tiny_samples[0], fm).matrix
    assert not np.allclose(values_afm, values_fm)
    assert np.abs(values_afm).max() <= 0.25 + 1e-12


# --------------------------------------------------------------------------------------
# Base cycles
# --------------------------------------------------------------------------------------


def test_ring_features_on_single_cycle():
    """On a 6-cycle, all pairs are co-cyclic, at distance 1, 2 or 3."""
    sample = make_sample(nx.cycle_graph(6), 0)
    matrix, labels = ring_features(sample.graph, sample.pair_index, max_ring_size=8)
    columns = dict(zip(labels, range(len(labels))))
    assert np.all(matrix[:, columns["in_ring_6"]] == 1.0)
    assert np.all(matrix[:, columns["n_rings_6"]] == 1.0)
    distances = matrix[:, columns["ring_dist_6"]]
    assert set(np.unique(distances)) == {1.0, 2.0, 3.0}
    position = _pair_position(sample, 0, 3)
    assert distances[position] == 3.0
    assert matrix[position, columns["ring_cos_6"]] == pytest.approx(-1.0)
    assert np.all(matrix[:, columns["ring_member_i"]] == 1.0)


def test_ring_features_flag_fused_rings(tiny_samples):
    """The merged graph carries a 6-cycle and a 5-cycle sharing an edge."""
    sample = tiny_samples[3]
    matrix, labels = ring_features(sample.graph, sample.pair_index, max_ring_size=8)
    columns = dict(zip(labels, range(len(labels))))
    assert matrix[:, columns["in_ring_6"]].sum() > 0
    assert matrix[:, columns["in_ring_5"]].sum() > 0
    assert matrix[:, columns["fused"]].sum() > 0
    # A pair straddling both cycles is co-cyclic in neither.
    across = _pair_position(sample, 0, 6)
    assert matrix[across, columns["in_ring_6"]] == 0.0
    assert matrix[across, columns["in_ring_5"]] == 0.0
    assert matrix[across, columns["fused"]] == 1.0


def test_ring_features_report_closed_walks():
    """On a triangle, (A^3)_ii = 2: closed walks count the cycles."""
    sample = make_sample(nx.cycle_graph(3), 0)
    matrix, labels = ring_features(sample.graph, sample.pair_index, max_ring_size=4)
    columns = dict(zip(labels, range(len(labels))))
    assert np.all(matrix[:, columns["closed_3_i"]] == 2.0)


def test_ring_aromatic_column_requires_bond_types():
    sample = make_sample(nx.cycle_graph(6), 0, dataset="MUTAG")
    matrix, labels = ring_features(
        sample.graph,
        sample.pair_index,
        max_ring_size=8,
        bond_types=sample.bond_types,
        aromatic_index=sample.aromatic_bond_index,
    )
    columns = dict(zip(labels, range(len(labels))))
    assert np.all(matrix[:, columns["aromatic"]] == 1.0)

    plain, _ = ring_features(sample.graph, sample.pair_index, max_ring_size=8)
    assert np.all(plain[:, columns["aromatic"]] == 0.0)


def test_interaction_basis_is_a_single_column_at_unit_nearest_neighbour(tiny_samples, context):
    """J~_ij equals 1 for the closest bonded pair, and decays as 1/d^6."""
    sample = tiny_samples[0]
    block = interaction_basis(sample, context)
    assert block.labels == ("interaction",)
    assert block.num_columns == 1
    assert block.matrix.max() == pytest.approx(1.0)
    assert block.matrix.min() > 0.0


def test_distance_basis_is_a_single_column_in_rnn_units(tiny_samples, context):
    sample = tiny_samples[0]
    block = distance_basis(sample, context)
    assert block.labels == ("distance",)
    assert block.matrix.min() == pytest.approx(1.0)


def test_interaction_and_distance_are_the_same_information_transformed(tiny_samples, context):
    """J~ = d^-6: the two columns are related by a decreasing bijection."""
    sample = tiny_samples[0]
    couplings = interaction_basis(sample, context).matrix[:, 0]
    distances = distance_basis(sample, context).matrix[:, 0]
    assert np.allclose(couplings, distances**-6.0)
    order_j = np.argsort(couplings)
    order_d = np.argsort(-distances)
    assert np.array_equal(order_j, order_d)


def test_interaction_column_matches_jpow_first_order(tiny_samples, context):
    """The minimal basis must coincide with the first order of jpow."""
    sample = tiny_samples[0]
    single = interaction_basis(sample, context).matrix[:, 0]
    first_power = interaction_power_basis(sample, context).select(1).matrix[:, 0]
    assert np.allclose(single, first_power)


@pytest.mark.parametrize("builder", [interaction_basis, distance_basis])
def test_minimal_geometric_bases_require_positions(builder, context):
    sample = make_sample(nx.cycle_graph(4), 0)
    sample.embedded_position = None
    with pytest.raises(ValueError, match="no atom positions"):
        builder(sample, context)
