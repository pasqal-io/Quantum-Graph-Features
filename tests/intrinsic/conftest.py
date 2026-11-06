import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from qgf.intrinsic.data_loading import orient
from qgf.intrinsic.pe_construction import QUANTUM_ATTRS, attach, classical_views

T = 6  # time steps of the synthetic quantum features

TOPOLOGIES = [
    (3, [(0, 1), (1, 2), (0, 2)]),  # triangle
    (5, [(0, 1), (1, 2), (2, 3), (1, 4)]),  # tree with branch
    (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),  # cycle-4
    (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]),  # hexagon + chord
    (4, [(0, 1), (0, 2), (0, 3), (1, 2)]),  # star-ish
    (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]),  # pentagon + chord
    (3, [(0, 1), (1, 2)]),  # path-3
    (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)]),  # two triangles bridged
    (4, [(0, 1), (1, 2), (2, 3)]),  # path-4
    (5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]),  # fan
]


def make_graph(num_nodes: int, edges, seed: int = 0) -> Data:
    """Graph with time-major random correlations (as stored in qgf data), oriented node-major."""
    rng = np.random.RandomState(seed)
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t()
    d = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        pos=torch.tensor(
            rng.rand(num_nodes, 2) * 20 + np.arange(num_nodes)[:, None] * 6, dtype=torch.float32
        ),
    )
    for group in ("fm", "afm"):
        ob = rng.randn(T, num_nodes).astype(np.float32)
        tb = rng.randn(T, num_nodes, num_nodes).astype(np.float32)
        tb = (tb + tb.transpose(0, 2, 1)) / 2
        tb[:, np.arange(num_nodes), np.arange(num_nodes)] = 0.0
        ob, tb = orient(ob, tb)
        ob_attr, tb_attr = QUANTUM_ATTRS[group]
        setattr(d, ob_attr, torch.tensor(np.ascontiguousarray(ob)))
        setattr(d, tb_attr, torch.tensor(np.ascontiguousarray(tb)))
    return d


@pytest.fixture
def small_graph() -> Data:
    return make_graph(3, [(0, 1), (1, 2), (0, 2)], seed=42)


@pytest.fixture
def medium_graph() -> Data:
    return make_graph(5, [(0, 1), (1, 2), (2, 3), (1, 4)], seed=43)


@pytest.fixture
def dataset() -> list[Data]:
    """10 synthetic graphs (enough for 5-fold CV) with raw classical views attached."""
    ds = [make_graph(n, e, seed=100 + i) for i, (n, e) in enumerate(TOPOLOGIES)]
    attach(ds, [classical_views(d) for d in ds])
    return ds
