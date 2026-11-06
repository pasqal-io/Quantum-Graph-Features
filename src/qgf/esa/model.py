"""ESA graph classifier with optional gated fusion of the quantum features (Table 4).

Port of the original ``Estimator`` (edge attention, binary classification), without the
PyTorch Lightning wrapper (training is in :mod:`qgf.esa.benchmark`). Module construction
order is kept so that a seed initialises the same parameters in the same order.

Edge representation (one token per directed edge ``i -> j``):

- ``concat``: ``node_edge_mlp([x_i, x_j, e_ij])`` with
  ``x = [one-hot atom (7) | rings (6) | quantum one-body (Q1) | RWSE encoding (24)]`` and
  ``e = [one-hot bond (4) | quantum two-body (Q2)]``;
- ``gated`` (App. E, Sec. 4.2): a structural path ``SmallMLP([s_i, s_j, bond])`` and a quantum
  path ``SmallMLP([q_i, q_j, q_ij])`` (both with LayerNorm) fused per dimension,
  ``h = LN(g * struct + (1 - g) * quantum)`` with ``g = sigmoid(W [struct, quantum] + b)``,
  ``b`` initialised to 2 (``g`` ~ 0.88: start from the structural model).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from qgf.esa.layers import ESA, SmallMLP, edge_adjacency_bias

RWSE_STEPS = 20
RWSE_DIM = 24


def nearest_multiple_of_8(n: int) -> int:
    return n if n % 8 == 0 else n + 8 - n % 8


class RWSEEncoder(nn.Module):
    """``BatchNorm1d(20)`` then ``Linear(20, 24)`` on the random-walk landing probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_norm = nn.BatchNorm1d(RWSE_STEPS)
        self.pe_encoder = nn.Linear(RWSE_STEPS, RWSE_DIM)

    def forward(self, rwse: torch.Tensor) -> torch.Tensor:
        return self.pe_encoder(self.raw_norm(rwse))


class GatedFusion(nn.Module):
    def __init__(self, struct_dim: int, quantum_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = max(num_layers, 2)
        self.struct_mlp = SmallMLP(
            struct_dim, hidden_dim, hidden_dim, num_layers=layers, use_ln=True
        )
        self.quantum_mlp = SmallMLP(
            quantum_dim, hidden_dim, hidden_dim, num_layers=layers, use_ln=True
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate.bias, 2.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, struct: torch.Tensor, quantum: torch.Tensor):
        s, q = self.struct_mlp(struct), self.quantum_mlp(quantum)
        g = torch.sigmoid(self.gate(torch.cat([s, q], dim=-1)))
        return self.norm(g * s + (1 - g) * q), g


class ESAClassifier(nn.Module):
    """Edge-set attention binary classifier (one logit per graph).

    Args:
        num_features: node feature dim (atoms + rings + quantum one-body), RWSE excluded.
        edge_dim: edge feature dim (bonds + quantum two-body).
        set_max_items: largest number of edges of a graph in the dataset.
        fusion: ``"concat"`` or ``"gated"``; ``quantum_node_dim`` / ``quantum_edge_dim``
            locate the quantum blocks (last ``quantum_node_dim`` node columns before RWSE,
            last ``quantum_edge_dim`` edge columns).
    """

    def __init__(
        self,
        num_features: int,
        edge_dim: int,
        set_max_items: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        graph_dim: int = 128,
        mlp_hidden_size: int = 512,
        num_mlp_layers: int = 2,
        num_pma_seeds: int = 32,
        fusion: str = "concat",
        quantum_node_dim: int = 0,
        quantum_edge_dim: int = 0,
        rwse: bool = True,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.set_max_items = set_max_items
        self.rwse_encoder = RWSEEncoder() if rwse else None
        node_dim = num_features + (RWSE_DIM if rwse else 0)
        self.fusion = fusion if (quantum_node_dim or quantum_edge_dim) else "concat"
        if self.fusion == "gated":
            struct_node = node_dim - quantum_node_dim
            struct_edge = max(edge_dim - quantum_edge_dim, 0)
            self.edge_fusion = GatedFusion(
                struct_node * 2 + struct_edge,
                quantum_node_dim * 2 + quantum_edge_dim,
                hidden_dim,
                num_layers=num_mlp_layers,
            )
            self.base_struct = num_features - quantum_node_dim  # atoms + rings
            self.quantum_node_dim = quantum_node_dim
            self.quantum_edge_dim = quantum_edge_dim
        else:
            self.node_edge_mlp = SmallMLP(
                node_dim * 2 + edge_dim, 128, hidden_dim, num_layers=num_mlp_layers
            )
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=1e-8)  # built, unused (as the original)
        layer_types = ["S"] + ["M"] * (num_layers - 3) + ["S", "P"]
        self.st_fast = ESA(
            num_outputs=num_pma_seeds,
            dim_output=graph_dim,
            dim_hidden=[hidden_dim] * num_layers,
            num_heads=[num_heads] * num_layers,
            layer_types=layer_types,
            mlp_hidden_size=mlp_hidden_size,
            num_mlp_layers=num_mlp_layers,
            use_mlp_ln=True,
        )
        # the original built a SmallMLP head and replaced it by a Linear: keep the RNG draws
        SmallMLP(graph_dim, 128, 1, num_layers=num_mlp_layers)
        self.output_mlp = nn.Linear(graph_dim, 1)
        self.last_gate_mean: float | None = None

    def edge_tokens(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """``[E, hidden]`` edge embeddings (concatenation or gated fusion)."""
        src, dst = x[edge_index[0]], x[edge_index[1]]
        if self.fusion != "gated":
            return self.node_edge_mlp(torch.cat([src, dst, edge_attr.float()], dim=1))
        b, q = self.base_struct, self.quantum_node_dim
        s_src = torch.cat([src[:, :b], src[:, b + q :]], dim=-1)
        s_dst = torch.cat([dst[:, :b], dst[:, b + q :]], dim=-1)
        e = edge_attr.float()
        split = e.shape[-1] - self.quantum_edge_dim
        struct = torch.cat([s_src, s_dst, e[:, :split]], dim=-1)
        quantum = torch.cat([src[:, b : b + q], dst[:, b : b + q], e[:, split:]], dim=-1)
        h, gate = self.edge_fusion(struct, quantum)
        self.last_gate_mean = gate.mean().item()
        return h

    def forward(self, batch) -> torch.Tensor:
        """``[B]`` logits."""
        x = batch.x.float()
        if self.rwse_encoder is not None:
            x = torch.cat([x, self.rwse_encoder(batch.pestat_RWSE)], dim=1)
        h = self.edge_tokens(x, batch.edge_index, batch.edge_attr)
        max_items = nearest_multiple_of_8(self.set_max_items + 1)
        edge_graph = batch.batch[batch.edge_index[0]]
        h, _ = to_dense_batch(h, edge_graph, fill_value=0, max_num_nodes=max_items)  # [B, L, D]
        bias = edge_adjacency_bias(batch.edge_index, batch.batch, h.shape[0], max_items)
        return torch.flatten(self.output_mlp(self.st_fast(h, bias)))
