"""Embedding layers and PE encoders."""

import torch
from torch import nn


class NodeEmbedding(nn.Module):
    """Embed categorical node features (atom types)."""

    def __init__(self, num_atom_types: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_atom_types, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N] - Node features (atom types)

        Returns:
            h: [N, D] - Node embeddings
        """
        return self.embed(x)  # [N, D]


class EdgeEmbedding(nn.Module):
    """Embed categorical edge features (bond types)."""

    def __init__(self, num_bond_types: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_bond_types, hidden_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: [E] - Edge features (bond types)

        Returns:
            h_edge: [E, D] - Edge embeddings
        """
        return self.embed(edge_attr)  # [E, D]


class AbsolutePEEncoder(nn.Module):
    """
    Encode absolute (node-level) positional encodings.
    Projects variable-dimension PE to fixed hidden_dim.

    Architecture: Linear(d_pe, D) -> ReLU -> Linear(D, D)
    """

    def __init__(self, pe_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.project = nn.Linear(pe_dim, hidden_dim)
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pe: [N, d_abs] - Absolute positional encodings

        Returns:
            h_pe: [N, D] - Encoded PE
        """
        h = self.project(pe)  # [N, d_abs] -> [N, D]
        return h + self.residual_block(h)  # Residual connection


class RelativePEEncoder(nn.Module):
    """
    Encode relative (pairwise) positional encodings.
    Projects variable-dimension PE to per-head attention bias.

    Architecture: Linear(d_rel, hidden) -> ReLU -> Linear(hidden, num_heads)
    Output is used as additive bias in attention: attn_ij += rel_pe_bias_ij
    """

    def __init__(self, pe_dim: int, num_heads: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.num_heads = num_heads
        self.encoder = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pe: [B, N, N, d_rel] - Relative PE (batched)

        Returns:
            bias: [B, N, N, H] - Attention bias per head
        """
        return self.encoder(pe)  # [B, N, N, d_rel] -> [B, N, N, H]
