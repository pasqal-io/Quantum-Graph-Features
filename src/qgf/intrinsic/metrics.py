"""Structural targets of the 15 intrinsic-benchmark tasks.

Ported as-is from the original internal implementation (node, edge, graph tasks); the chemical
(rdkit) tasks of the internal benchmark are not part of the paper and are dropped.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np


def _ensure_connected(G: nx.Graph) -> nx.Graph:
    """Return largest connected component if graph is disconnected."""
    if nx.is_connected(G):
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


# ── node ──────────────────────────────────────────────────────────────────


def degree_targets(G: nx.Graph) -> np.ndarray:
    return np.array([G.degree(n) for n in sorted(G.nodes())], dtype=np.float64)


def betweenness_targets(G: nx.Graph) -> np.ndarray:
    bc = nx.betweenness_centrality(G)
    return np.array([bc[n] for n in sorted(G.nodes())], dtype=np.float64)


def bridge_node_targets(G: nx.Graph) -> np.ndarray:
    bridge_edges = list(nx.bridges(G))
    bridge_nodes = set(itertools.chain.from_iterable(bridge_edges))
    return np.array([n in bridge_nodes for n in sorted(G.nodes())], dtype=bool)


def eccentricity_targets(G: nx.Graph) -> np.ndarray:
    G_conn = _ensure_connected(G)
    ecc = nx.eccentricity(G_conn)
    return np.array([ecc[n] for n in sorted(G_conn.nodes())], dtype=np.float64)


def smallest_cycle_node_targets(G: nx.Graph) -> np.ndarray:
    """Size of smallest cycle containing each node (0 if not in any cycle)."""
    nodes = sorted(G.nodes())
    result = np.zeros(len(nodes), dtype=np.float64)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    for cycle in nx.cycle_basis(G):
        size = len(cycle)
        for n in cycle:
            idx = node_to_idx[n]
            if result[idx] == 0 or size < result[idx]:
                result[idx] = float(size)
    return result


# ── edge (all upper-triangular node pairs, non-edges included) ────────────


def link_prediction_targets(G: nx.Graph, num_nodes: int) -> np.ndarray:
    """Binary: 1 if edge exists, 0 otherwise. Upper-triangular order."""
    edge_set = set()
    for u, v in G.edges():
        edge_set.add((min(u, v), max(u, v)))
    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    return np.array(
        [(int(triu_i[idx]), int(triu_j[idx])) in edge_set for idx in range(triu_i.size)],
        dtype=np.float64,
    )


def edge_betweenness_targets(G: nx.Graph, num_nodes: int) -> np.ndarray:
    """Edge betweenness centrality. Non-edges -> 0.0. Upper-triangular order."""
    raw_ebc = nx.edge_betweenness_centrality(G, normalized=True)
    ebc_map: dict[tuple[int, int], float] = {}
    for (u, v), val in raw_ebc.items():
        key = (min(u, v), max(u, v))
        ebc_map[key] = float(val)

    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    arr = np.zeros(triu_i.size, dtype=np.float64)
    for idx in range(triu_i.size):
        key = (int(triu_i[idx]), int(triu_j[idx]))
        if key in ebc_map:
            arr[idx] = ebc_map[key]
    return arr


def bridge_edge_targets(G: nx.Graph, num_nodes: int) -> np.ndarray:
    """Binary: 1 if edge is a bridge, 0 otherwise. Upper-triangular order."""
    bridge_set = set()
    for u, v in nx.bridges(G):
        bridge_set.add((min(u, v), max(u, v)))

    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    return np.array(
        [(int(triu_i[idx]), int(triu_j[idx])) in bridge_set for idx in range(triu_i.size)],
        dtype=np.float64,
    )


def smallest_cycle_edge_targets(G: nx.Graph, num_nodes: int) -> np.ndarray:
    """Size of smallest cycle containing each edge (0 if not in any cycle)."""
    edge_min_cycle: dict[tuple[int, int], float] = {}
    for cycle in nx.cycle_basis(G):
        size = len(cycle)
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            key = (min(u, v), max(u, v))
            if key not in edge_min_cycle or size < edge_min_cycle[key]:
                edge_min_cycle[key] = float(size)

    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    arr = np.zeros(triu_i.size, dtype=np.float64)
    for idx in range(triu_i.size):
        key = (int(triu_i[idx]), int(triu_j[idx]))
        if key in edge_min_cycle:
            arr[idx] = edge_min_cycle[key]
    return arr


def local_edge_connectivity_targets(G: nx.Graph, num_nodes: int) -> np.ndarray:
    """Local edge connectivity between each node pair. Upper-triangular order."""
    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    arr = np.zeros(triu_i.size, dtype=np.float64)
    for idx in range(triu_i.size):
        u, v = int(triu_i[idx]), int(triu_j[idx])
        arr[idx] = float(nx.connectivity.local_edge_connectivity(G, u, v))
    return arr


# ── graph ─────────────────────────────────────────────────────────────────


def num_cycles_target(G: nx.Graph) -> float:
    return float(len(nx.cycle_basis(G)))


def graph_diameter_target(G: nx.Graph) -> float:
    return float(nx.diameter(_ensure_connected(G)))


def graph_density_target(G: nx.Graph) -> float:
    return float(nx.density(G))


def wiener_index_target(G: nx.Graph) -> float:
    """Normalized Wiener index: average shortest path distance between all node pairs."""
    G_conn = _ensure_connected(G)
    N = G_conn.number_of_nodes()
    if N <= 1:
        return 0.0
    w = nx.wiener_index(G_conn)
    num_pairs = N * (N - 1) / 2
    return float(w / num_pairs)


def algebraic_connectivity_target(G: nx.Graph) -> float:
    return float(nx.algebraic_connectivity(_ensure_connected(G)))


# category -> {task -> {func, classification}}
BENCHMARK_REGISTRY: dict[str, dict] = {
    "node_prediction": {
        "degree_prediction": {"func": degree_targets, "classification": False},
        "betweenness_prediction": {"func": betweenness_targets, "classification": False},
        "bridge_node_prediction": {"func": bridge_node_targets, "classification": True},
        "node_eccentricity": {"func": eccentricity_targets, "classification": False},
        "smallest_cycle_node": {"func": smallest_cycle_node_targets, "classification": False},
    },
    "edge_prediction": {
        "link_prediction": {"func": link_prediction_targets, "classification": True},
        "edge_betweenness": {"func": edge_betweenness_targets, "classification": False},
        "smallest_cycle_edge": {"func": smallest_cycle_edge_targets, "classification": False},
        "is_bridge_edge": {"func": bridge_edge_targets, "classification": True},
        "local_edge_connectivity": {
            "func": local_edge_connectivity_targets,
            "classification": False,
        },
    },
    "graph_prediction": {
        "num_cycles": {"func": num_cycles_target, "classification": False},
        "graph_diameter": {"func": graph_diameter_target, "classification": False},
        "graph_density": {"func": graph_density_target, "classification": False},
        "wiener_index": {"func": wiener_index_target, "classification": False},
        "algebraic_connectivity": {"func": algebraic_connectivity_target, "classification": False},
    },
}

TASKS: list[str] = [b for cat in BENCHMARK_REGISTRY.values() for b in cat]


def get_category(benchmark_name: str) -> str:
    """Return the category for a benchmark name."""
    for category, benchmarks in BENCHMARK_REGISTRY.items():
        if benchmark_name in benchmarks:
            return category
    raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {TASKS}")


def compute_targets(benchmark: str, graphs: list[nx.Graph]):
    """Targets for every graph: list of per-entity arrays (node/edge) or a 1D array (graph)."""
    category = get_category(benchmark)
    func = BENCHMARK_REGISTRY[category][benchmark]["func"]
    if category == "node_prediction":
        return [func(g) for g in graphs]
    if category == "edge_prediction":
        return [func(g, num_nodes=g.number_of_nodes()) for g in graphs]
    return np.array([func(g) for g in graphs])
