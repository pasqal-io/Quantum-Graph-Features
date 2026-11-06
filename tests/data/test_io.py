import pytest
import torch
from torch_geometric.data import Data

from qgf.data.io import FORMAT_VERSION, load_dataset, save_dataset


def _graph(n: int) -> Data:
    return Data(
        x=torch.randn(n, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([1]),
        onebody_features=torch.rand(4, n),
        twobody_features=torch.rand(4, n, n),
    )


def test_roundtrip(tmp_path):
    graphs = [_graph(3), _graph(5)]
    meta = {"name": "toy", "times_ns": [100, 200, 300, 400]}
    path = tmp_path / "toy.pt"
    save_dataset(path, graphs, meta)

    loaded, loaded_meta = load_dataset(path)

    assert loaded_meta == meta
    assert len(loaded) == 2
    for a, b in zip(graphs, loaded):
        assert set(a.keys()) == set(b.keys())
        for k in set(a.keys()):
            assert torch.equal(a[k], b[k])


def test_file_is_weights_only_loadable(tmp_path):
    path = tmp_path / "toy.pt"
    save_dataset(path, [_graph(2)], {"name": "toy"})
    blob = torch.load(path, weights_only=True)
    assert blob["format_version"] == FORMAT_VERSION


def test_rejects_non_tensor_attribute(tmp_path):
    g = _graph(2)
    g.label_name = "not a tensor"
    with pytest.raises(TypeError, match="label_name"):
        save_dataset(tmp_path / "bad.pt", [g], {})
