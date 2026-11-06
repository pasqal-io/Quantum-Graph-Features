"""Message-passing / attention backbones with quantum and classical encodings (Tables 2-3).

Port of the original internal ``GCNorGATwithPE``, restricted to the five backbones of the
paper: GCN, GAT, GINE, Transformer (``TransformerConv``) and General (``GeneralConv``).
Module construction order is kept, so a seed initialises the same parameters in the same
order as the original.

Feature injection (App. E.2):

- atoms: learned embedding (dimension 75); one-body encodings ``[N, abs_dim]`` are
  concatenated to it before the first layer;
- bonds: learned embedding (dimension 50);
- two-body encodings ``[N, N, rel_dim]``:
  - GCN (edge weights): bond-embedding matrices are stacked under the two-body channels,
    reduced to one scalar per node pair by a learned linear combination
    (:class:`LinearCorrelationEncoder`) and read on the edges; without two-body features
    the bond embedding alone goes through the encoder;
  - edge-attribute backbones: the two-body vector of each edge is concatenated with its
    bond embedding and passed as ``edge_attr``.

Edge weights are clamped to [1e-8, 100] and, with ``edge_weight_shift`` (used for GCN),
first shifted per graph so that the smallest is ``1e-3`` (no negative degrees in the
``D^-1/2 A D^-1/2`` normalisation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

EDGE_ATTR_MODELS = frozenset({"GAT", "GINE", "Transformer", "General", "GATv2", "GEN"})
EDGE_WEIGHT_MODELS = frozenset({"GCN"})
MODELS = ("GCN", "GAT", "GINE", "Transformer", "General")  # backbones of the paper tables
EXTRA_MODELS = ("GATv2", "GEN")  # also in the strong-baseline runs (experiments/non_unit_disk)
ALL_MODELS = MODELS + EXTRA_MODELS

ATOM_EMBEDDING_DIM = 75
BOND_EMBEDDING_DIM = 50


class LinearCorrelationEncoder(nn.Module):
    """``[M, N, N] -> [N, N]``: learned linear combination of M pair matrices
    (coefficients initialised to ``1 / M``)."""

    def __init__(self, num_matrices: int) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.ones(num_matrices) / num_matrices)

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        return torch.sum(matrices * self.coefficients.view(-1, 1, 1), dim=0)  # [N, N]


class GNNWithPE(nn.Module):
    """Graph-level GNN consuming ``batch.onebody_features`` (list of ``[N_i, abs_dim]``)
    and ``batch.twobody_features`` (list of ``[N_i, N_i, rel_dim]``)."""

    def __init__(
        self,
        model_type: str,
        num_atom_types: int,
        num_bond_types: int,
        gnn_layers: list[int],
        mlp_layers: list[int],
        abs_pe_dim: int = 0,
        rel_pe_dim: int = 0,
        output_dim: int = 1,
        pooling: str = "mean",
        dropout: float = 0.1,
        num_heads: int = 1,
        layer_norm: bool = True,
        edge_weight_clamp: tuple[float, float] | None = (1e-8, 100.0),
        edge_weight_shift: bool = False,
        edge_weight_shift_eps: float = 1e-3,
        bond_features: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in ALL_MODELS:
            raise ValueError(f"model_type must be one of {ALL_MODELS}, got {model_type!r}")
        self.model_type = model_type
        self.abs_pe_dim = abs_pe_dim
        self.rel_pe_dim = rel_pe_dim
        self.num_bond_types = num_bond_types
        self.pooling = pooling
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.edge_weight_clamp = edge_weight_clamp
        self.edge_weight_shift = edge_weight_shift
        self.edge_weight_shift_eps = edge_weight_shift_eps
        self.bond_features = bond_features
        is_edge_weight = model_type in EDGE_WEIGHT_MODELS
        bond_dim = BOND_EMBEDDING_DIM if bond_features else 0

        self.node_embedding = nn.Embedding(num_atom_types, ATOM_EMBEDDING_DIM)
        if is_edge_weight and rel_pe_dim + bond_dim > 0:
            self.twobody_encoder = LinearCorrelationEncoder(rel_pe_dim + bond_dim)
        if bond_features:
            self.bond_embedding = nn.Embedding(num_bond_types, BOND_EMBEDDING_DIM)
        edge_dim = None if is_edge_weight or rel_pe_dim + bond_dim == 0 else rel_pe_dim + bond_dim

        self.convs = nn.ModuleList()
        prev = ATOM_EMBEDDING_DIM + abs_pe_dim
        for hidden in gnn_layers:
            self.convs.append(_conv(model_type, prev, hidden, edge_dim, num_heads))
            prev = hidden
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(d) for d in gnn_layers])

        mlp: list[nn.Module] = []
        for hidden in mlp_layers:
            mlp += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        mlp.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*mlp)

    # ------------------------------------------------------------------
    # pair features on the edges
    # ------------------------------------------------------------------

    def _bonds(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """``[E]`` or ``[E, 1]`` bond indices -> ``[E, BOND_EMBEDDING_DIM]``."""
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        return self.bond_embedding(torch.clamp(edge_attr.long(), 0, self.num_bond_types - 1))

    @staticmethod
    def _edge_ptr(batch: Batch) -> torch.Tensor:
        """``[B + 1]`` edge offsets of each graph (edges are stored graph by graph)."""
        graph_of_edge = batch.batch[batch.edge_index[0]]  # [E]
        n_graphs = batch.ptr.shape[0] - 1
        ptr = torch.zeros(n_graphs + 1, dtype=torch.long, device=graph_of_edge.device)
        if graph_of_edge.numel() > 0:
            counts = torch.zeros(n_graphs, dtype=torch.long, device=graph_of_edge.device)
            counts.scatter_add_(0, graph_of_edge, torch.ones_like(graph_of_edge))
            torch.cumsum(counts, dim=0, out=ptr[1:])
        return ptr

    def _per_graph(self, batch: Batch, twobody: list[torch.Tensor] | None) -> torch.Tensor | None:
        """Edge weights ``[E]`` (GCN) or edge attributes ``[E, rel_dim + 50]`` (``None``:
        no pair features; GCN then uses unit weights).

        As the original, bonds are embedded graph by graph when two-body features are present
        and for the whole batch otherwise: the embedding gradient is summed in that order, so
        this keeps trials bit-identical to the published ones.
        """
        if not twobody and not self.bond_features:
            if self.model_type in EDGE_WEIGHT_MODELS:
                return torch.ones(batch.edge_index.shape[1], device=batch.edge_index.device)
            return None
        if not self.bond_features:  # two-body only
            return self._twobody_only(batch, twobody)
        bonds = None if twobody else self._bonds(batch.edge_attr)  # [E, 50]
        edge_ptr = self._edge_ptr(batch)
        out = []
        for g in range(batch.ptr.shape[0] - 1):
            n0, n1 = batch.ptr[g].item(), batch.ptr[g + 1].item()
            e0, e1 = edge_ptr[g].item(), edge_ptr[g + 1].item()
            src, dst = batch.edge_index[:, e0:e1] - n0
            bond = self._bonds(batch.edge_attr[e0:e1]) if twobody else bonds[e0:e1]
            if self.model_type in EDGE_WEIGHT_MODELS:
                n = n1 - n0
                dtype = twobody[g].dtype if twobody else bond.dtype
                bond_mats = torch.zeros(n, n, bond.shape[-1], device=bond.device, dtype=dtype)
                bond_mats[src, dst] = bond.to(dtype)
                mats = torch.cat([twobody[g], bond_mats], dim=-1) if twobody else bond_mats
                out.append(self.twobody_encoder(mats.permute(2, 0, 1).contiguous())[src, dst])
            elif twobody:
                tb = twobody[g][src, dst]  # [E_g, rel_dim]
                out.append(torch.cat([tb, bond.to(tb.dtype)], dim=-1))
            else:
                out.append(bond)
        return torch.cat(out, dim=0)

    def _twobody_only(self, batch: Batch, twobody: list[torch.Tensor]) -> torch.Tensor:
        edge_ptr = self._edge_ptr(batch)
        out = []
        for g in range(batch.ptr.shape[0] - 1):
            n0 = batch.ptr[g].item()
            e0, e1 = edge_ptr[g].item(), edge_ptr[g + 1].item()
            src, dst = batch.edge_index[:, e0:e1] - n0
            if self.model_type in EDGE_WEIGHT_MODELS:
                out.append(self.twobody_encoder(twobody[g].permute(2, 0, 1).contiguous())[src, dst])
            else:
                out.append(twobody[g][src, dst])  # [E_g, rel_dim]
        return torch.cat(out, dim=0)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.node_embedding(batch.x.squeeze(-1))  # [sum N, 75]
        onebody = getattr(batch, "onebody_features", None)
        if self.abs_pe_dim > 0 and onebody:
            x = torch.cat([x, torch.cat(onebody, dim=0).to(x.device)], dim=1)
        twobody = getattr(batch, "twobody_features", None) if self.rel_pe_dim > 0 else None

        pair = self._per_graph(batch, twobody)
        edge_weight = edge_attr = None
        if self.model_type in EDGE_WEIGHT_MODELS:
            edge_weight = pair
            encoded = twobody or self.bond_features  # unit weights are not clamped
            if self.edge_weight_clamp is not None and encoded:
                edge_weight = torch.clamp(edge_weight, *self.edge_weight_clamp)
            if self.edge_weight_shift:
                gid = batch.batch[batch.edge_index[0]]  # [E]
                edge_weight = shift_per_graph(edge_weight, gid, self.edge_weight_shift_eps)
        else:
            edge_attr = pair

        for i, conv in enumerate(self.convs):
            if edge_weight is not None:
                x = F.relu(conv(x, batch.edge_index, edge_weight=edge_weight))
            elif edge_attr is not None:
                x = F.relu(conv(x, batch.edge_index, edge_attr=edge_attr))
            else:
                x = F.relu(conv(x, batch.edge_index))
            if self.layer_norm:
                x = self.layer_norms[i](x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        pool = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}
        return self.mlp(pool[self.pooling](x, batch.batch))


def shift_per_graph(weight: torch.Tensor, graph_id: torch.Tensor, eps: float) -> torch.Tensor:
    """Shift edge weights ``[E]`` so that each graph's smallest weight is ``eps``."""
    big = torch.full_like(weight, float("inf"))
    gmin = big.scatter_reduce(0, graph_id, weight, reduce="amin")  # [E], first B entries used
    return weight - gmin[graph_id] + eps


def _conv(model_type: str, d_in: int, d_out: int, edge_dim: int | None, heads: int) -> nn.Module:
    if model_type == "GCN":
        return pyg_nn.GCNConv(d_in, d_out, normalize=True)
    if model_type == "GAT":
        return pyg_nn.GATConv(d_in, d_out, heads=heads, edge_dim=edge_dim, concat=False)
    if model_type == "GATv2":
        return pyg_nn.GATv2Conv(d_in, d_out, edge_dim=edge_dim)
    if model_type == "Transformer":
        return pyg_nn.TransformerConv(d_in, d_out, edge_dim=edge_dim)
    if model_type == "GEN":
        return pyg_nn.GENConv(d_in, d_out, edge_dim=edge_dim)
    if model_type == "GINE":
        mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))
        return pyg_nn.GINEConv(mlp, edge_dim=edge_dim)
    return pyg_nn.GeneralConv(d_in, d_out, in_edge_channels=edge_dim)  # General
