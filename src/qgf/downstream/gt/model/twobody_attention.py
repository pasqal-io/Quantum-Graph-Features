"""Twobody PE injection modules (all SDPA-compatible).

Strategies for injecting twobody features beyond the baseline RelativePEEncoder:
- EdgeFusionAttention: fuse twobody with edge embeddings before bias projection
- DeepBiasEncoder: deeper MLP to produce per-head attention bias
- NodeEnrichment: mean+max pool twobody per node, enrich node embeddings
- TwobodyToEdgeProjector: project twobody to edge-embedding space (used by edge strategies)
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Reusable projector
# ---------------------------------------------------------------------------


class TwobodyToEdgeProjector(nn.Module):
    """Project twobody features to edge-embedding space.

    Architecture: Linear(d_rel, D) → ReLU → Linear(D, D)
    """

    def __init__(self, rel_pe_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rel_pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, twobody: torch.Tensor) -> torch.Tensor:
        """
        Args:
            twobody: [B, N, N, d_rel]

        Returns:
            projected: [B, N, N, D]
        """
        return self.mlp(twobody)  # [B, N, N, d_rel] -> [B, N, N, D]


# ---------------------------------------------------------------------------
# Deep Bias Encoder
# ---------------------------------------------------------------------------


class DeepBiasEncoder(nn.Module):
    """Deeper MLP to project twobody features to per-head attention bias.

    Architecture: Linear(d_rel, D) → ReLU → Linear(D, D) → ReLU → Linear(D, H)
    Output used as additive bias alongside the edge bias in standard attention.
    """

    def __init__(self, rel_pe_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(rel_pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, twobody: torch.Tensor) -> torch.Tensor:
        """
        Args:
            twobody: [B, N, N, d_rel]

        Returns:
            bias: [B, N, N, H] — additive attention bias per head
        """
        return self.encoder(twobody)  # [B, N, N, d_rel] -> [B, N, N, H]


# ---------------------------------------------------------------------------
# Node Enrichment
# ---------------------------------------------------------------------------


class NodeEnrichment(nn.Module):
    """Enrich node embeddings with aggregated twobody features.

    Aggregates twobody features per node using mean+max pooling over neighbors,
    then projects to hidden_dim and adds to node embeddings.

    agg_i = [mean_j(twobody_ij) ; max_j(twobody_ij)]  # [B, N, 2*d_rel]
    out = MLP(agg_i)                                     # [B, N, D]
    """

    def __init__(self, rel_pe_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * rel_pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, twobody: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            twobody: [B, N, N, d_rel]
            padding_mask: [B, N] or None — True = valid node

        Returns:
            enrichment: [B, N, D] — additive enrichment for node embeddings
        """
        # Mask out padding nodes before pooling to avoid polluting stats
        if padding_mask is not None:
            # Expand mask to [B, 1, N, 1] to mask the j (neighbor) dimension
            mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            twobody_masked = twobody.masked_fill(~mask, 0.0)  # [B, N, N, d_rel]
            # For max pooling, use -inf for masked positions
            twobody_for_max = twobody.masked_fill(~mask, float("-inf"))  # [B, N, N, d_rel]
        else:
            twobody_masked = twobody
            twobody_for_max = twobody

        # Mean pooling over neighbor axis (dim=2)
        if padding_mask is not None:
            # Count valid neighbors per node
            valid_count = padding_mask.float().sum(dim=1, keepdim=True)  # [B, 1]
            valid_count = valid_count.unsqueeze(-1).clamp(min=1.0)  # [B, 1, 1]
            mean_agg = twobody_masked.sum(dim=2) / valid_count  # [B, N, d_rel]
        else:
            mean_agg = twobody_masked.mean(dim=2)  # [B, N, d_rel]

        # Max pooling over neighbor axis (dim=2)
        max_agg = twobody_for_max.max(dim=2).values  # [B, N, d_rel]

        # Concatenate mean and max
        agg = torch.cat([mean_agg, max_agg], dim=-1)  # [B, N, 2*d_rel]

        return self.mlp(agg)  # [B, N, D]
