"""Graph pooling layers."""

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool


class MeanPooling(nn.Module):
    """Mean pooling over nodes → graph-level representation."""

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(x, batch)  # [N, D] -> [B, D]


class SumPooling(nn.Module):
    """Sum pooling over nodes → graph-level representation.

    More expressive than mean pooling for graphs of varying size:
    two graphs with different numbers of nodes but identical per-node
    averages will produce different sum representations.
    """

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_add_pool(x, batch)  # [N, D] -> [B, D]
