"""Fixtures for the reconstructibility experiment: small synthetic graphs.

No test accesses the qcorr datasets: everything runs on graphs of a few nodes,
so it stays executable off the cluster. Runs on MUTAG and ZINC go through Slurm
(``scripts/submit_expressivity.sh``).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qgf.reconstruction import registry  # noqa: F401  populates the basis registry
from qgf.reconstruction.config import R_NN_UM, BasisContext
from qgf.reconstruction.data import GraphSample

TIMES_NS = np.array([100.0, 200.0, 300.0, 400.0])


def scaled_layout(graph: nx.Graph, r_nn: float = R_NN_UM) -> np.ndarray:
    """2D layout whose shortest bonded distance equals ``r_nn``, like the
    registries."""
    positions = np.asarray(list(nx.kamada_kawai_layout(graph).values()))
    adjacency = nx.to_numpy_array(graph)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    rows, cols = np.triu_indices(graph.number_of_nodes(), k=1)
    bonded = distances[rows, cols][adjacency[rows, cols] > 0]
    return positions * (r_nn / bonded.min())


def make_sample(
    graph: nx.Graph,
    graph_id: int,
    correlations: np.ndarray | None = None,
    dataset: str = "MUTAG",
) -> GraphSample:
    """Build a GraphSample from a networkx graph."""
    graph = nx.convert_node_labels_to_integers(graph)
    num_nodes = graph.number_of_nodes()
    adjacency = nx.to_numpy_array(graph)
    if correlations is None:
        # Deterministic, symmetric target: 0.1 A_ij + 0.02 (A^2)_ij per time step.
        squared = adjacency @ adjacency
        np.fill_diagonal(squared, 0.0)
        correlations = np.stack(
            [(0.1 * adjacency + 0.02 * squared) * (1.0 + 0.1 * step) for step in range(4)],
        )
    return GraphSample(
        graph_id=graph_id,
        adjacency=adjacency.astype(np.int8),
        correlations=correlations,
        times_ns=TIMES_NS,
        embedded_position=scaled_layout(graph),
        node_types=np.zeros(num_nodes, dtype=np.int16),
        bond_types=np.where(adjacency > 0, 0, -1).astype(np.int16),
        dataset=dataset,
    )


@pytest.fixture
def tiny_graphs() -> list[nx.Graph]:
    """A cycle, a path, a lollipop and two merged cycles."""
    fused = nx.Graph()
    fused.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (3, 6), (6, 7), (7, 8), (8, 4)],
    )
    return [nx.cycle_graph(6), nx.path_graph(5), nx.lollipop_graph(4, 2), fused]


@pytest.fixture
def tiny_samples(tiny_graphs: list[nx.Graph]) -> list[GraphSample]:
    return [make_sample(graph, index) for index, graph in enumerate(tiny_graphs)]


@pytest.fixture
def context() -> BasisContext:
    return BasisContext(k_max=4, times_ns=tuple(TIMES_NS))
