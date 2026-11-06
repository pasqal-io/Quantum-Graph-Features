"""Multi-head attention with edge and PE bias using SDPA."""

import torch
import torch.nn.functional as F
from torch import nn


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with GATED edge and relative PE bias.

    Uses tanh gating to prevent bias explosion and keep values in a comparable
    range to QK^T/sqrt(d). The gating formula is:

        bias_final = tanh(Linear(h_edge)) * λ_learnable

    Where λ is a per-head learnable scale parameter initialized to 1.0.

    This prevents the structural bias from dominating the content signal (QK^T),
    which can lead to memorization rather than generalization.

    Attention formula:
        attn_score_ij = (Q_i · K_j) / sqrt(d_head) + gated_edge_bias_ij + gated_rel_pe_bias_ij
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias_scale_init: float = 10.0,
        edge_bias_dim: int = 0,
    ) -> None:
        """
        Initialize GatedMultiHeadAttention.

        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias_scale_init: Initial value for learnable bias scale λ
            edge_bias_dim: Dimension of edge embeddings for bias projection (0 = disabled)
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias projection (optional: disabled when edge_bias_dim=0)
        self.edge_bias_proj: nn.Linear | None = (
            nn.Linear(edge_bias_dim, num_heads) if edge_bias_dim > 0 else None
        )

        # Learnable scale factors for gated biases (one per head)
        self.edge_bias_scale = nn.Parameter(torch.full((num_heads,), bias_scale_init))  # [H]
        self.rel_pe_bias_scale = nn.Parameter(torch.full((num_heads,), bias_scale_init))  # [H]

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_embed_dense: torch.Tensor,
        rel_pe_bias: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Node features (dense batched)
            edge_embed_dense: [B, N, N, D] - Edge embeddings (dense)
            rel_pe_bias: [B, N, N, H] or None - Relative PE bias per head
            padding_mask: [B, N] or None - True = valid node, False = padding

        Returns:
            out: [B, N, D] - Updated node features
        """
        B, N, D = x.shape
        H = self.num_heads
        d_head = self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(x)  # [B, N, D]
        v = self.v_proj(x)  # [B, N, D]

        # Reshape for multi-head: [B, N, D] -> [B, N, H, d_head] -> [B, H, N, d_head]
        q = q.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]
        k = k.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]
        v = v.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]

        # Build GATED attention bias
        attn_bias = self._build_gated_attention_bias(
            edge_embed_dense, rel_pe_bias, padding_mask
        )  # [B, H, N, N]

        # Apply SDPA with combined bias
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, H, N, d_head]

        # Reshape back: [B, H, N, d_head] -> [B, N, H, d_head] -> [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # Output projection
        out = self.out_proj(out)  # [B, N, D]

        return out

    def _build_gated_attention_bias(
        self,
        edge_embed_dense: torch.Tensor | None,
        rel_pe_bias: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Build GATED combined attention bias from edges, PE, and padding.

        Uses tanh gating to bound bias values: bias = tanh(raw_bias) * λ

        Args:
            edge_embed_dense: [B, N, N, D] or None - Dense edge embeddings
            rel_pe_bias: [B, N, N, H] or None - Relative PE bias per head
            padding_mask: [B, N] or None - True = valid, False = padding

        Returns:
            attn_bias: [B, H, N, N] - Combined gated attention bias
        """
        attn_bias: torch.Tensor | None = None

        # Gated edge bias (when edge_bias_proj exists)
        if self.edge_bias_proj is not None and edge_embed_dense is not None:
            raw_edge_bias = self.edge_bias_proj(edge_embed_dense)  # [B, N, N, H]
            edge_scale = self.edge_bias_scale.view(1, 1, 1, -1)  # [1, 1, 1, H]
            gated_edge_bias = torch.tanh(raw_edge_bias) * edge_scale  # [B, N, N, H]
            attn_bias = gated_edge_bias.permute(0, 3, 1, 2)  # [B, H, N, N]

        # Add GATED relative PE bias (dense)
        if rel_pe_bias is not None:
            rel_pe_scale = self.rel_pe_bias_scale.view(1, 1, 1, -1)  # [1, 1, 1, H]
            gated_rel_pe_bias = torch.tanh(rel_pe_bias) * rel_pe_scale  # [B, N, N, H]
            gated_rel_pe_transposed = gated_rel_pe_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
            if attn_bias is None:
                attn_bias = gated_rel_pe_transposed
            else:
                attn_bias = attn_bias + gated_rel_pe_transposed  # [B, H, N, N]

        # If no bias sources, create zero bias for padding mask application
        if attn_bias is None:
            if padding_mask is None:
                return torch.zeros(
                    1, device=rel_pe_bias.device if rel_pe_bias is not None else "cpu"
                )
            B, N = padding_mask.shape
            attn_bias = torch.zeros(
                B, self.num_heads, N, N, device=padding_mask.device
            )  # [B, H, N, N]

        # Apply padding mask
        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_bias = attn_bias.masked_fill(~pad_mask, float("-inf"))  # [B, H, N, N]
            pad_mask_row = padding_mask.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1]
            attn_bias = attn_bias.masked_fill(~pad_mask_row, float("-inf"))  # [B, H, N, N]

        return attn_bias


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with edge and relative PE bias.

    Uses PyTorch's scaled_dot_product_attention (SDPA) for efficient computation
    with automatic FlashAttention/memory-efficient backend selection.

    Attention formula:
        attn_score_ij = (Q_i · K_j) / sqrt(d_head) + edge_bias_ij + rel_pe_bias_ij
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        edge_bias_dim: int = 0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias projection (optional: disabled when edge_bias_dim=0)
        self.edge_bias_proj: nn.Linear | None = (
            nn.Linear(edge_bias_dim, num_heads) if edge_bias_dim > 0 else None
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_embed_dense: torch.Tensor,
        rel_pe_bias: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Node features (dense batched)
            edge_embed_dense: [B, N, N, D] - Edge embeddings (dense)
            rel_pe_bias: [B, N, N, H] or None - Relative PE bias per head
            padding_mask: [B, N] or None - True = valid node, False = padding

        Returns:
            out: [B, N, D] - Updated node features
        """
        B, N, D = x.shape
        H = self.num_heads
        d_head = self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(x)  # [B, N, D]
        v = self.v_proj(x)  # [B, N, D]

        # Reshape for multi-head: [B, N, D] -> [B, N, H, d_head] -> [B, H, N, d_head]
        q = q.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]
        k = k.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]
        v = v.view(B, N, H, d_head).transpose(1, 2)  # [B, H, N, d_head]

        # Build attention bias
        attn_bias = self._build_attention_bias(
            edge_embed_dense, rel_pe_bias, padding_mask
        )  # [B, H, N, N]

        # Apply SDPA with combined bias
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, H, N, d_head]

        # Reshape back: [B, H, N, d_head] -> [B, N, H, d_head] -> [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # Output projection
        out = self.out_proj(out)  # [B, N, D]

        return out

    def _build_attention_bias(
        self,
        edge_embed_dense: torch.Tensor | None,
        rel_pe_bias: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Build combined attention bias from edges, PE, and padding.

        Args:
            edge_embed_dense: [B, N, N, D] or None - Dense edge embeddings
            rel_pe_bias: [B, N, N, H] or None - Relative PE bias per head
            padding_mask: [B, N] or None - True = valid, False = padding

        Returns:
            attn_bias: [B, H, N, N] - Combined attention bias
                       0 = attend, -inf = masked out
        """
        attn_bias: torch.Tensor | None = None

        # Project edge embeddings to per-head bias (when edge_bias_proj exists)
        if self.edge_bias_proj is not None and edge_embed_dense is not None:
            edge_bias = self.edge_bias_proj(edge_embed_dense)  # [B, N, N, H]
            attn_bias = edge_bias.permute(0, 3, 1, 2)  # [B, H, N, N]

        # Add relative PE bias (dense)
        if rel_pe_bias is not None:
            rel_pe_bias_transposed = rel_pe_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
            if attn_bias is None:
                attn_bias = rel_pe_bias_transposed
            else:
                attn_bias = attn_bias + rel_pe_bias_transposed  # [B, H, N, N]

        # If no bias sources, create zero bias for padding mask application
        if attn_bias is None:
            # Infer shape from padding_mask or return None if no mask
            if padding_mask is None:
                return torch.zeros(
                    1, device=rel_pe_bias.device if rel_pe_bias is not None else "cpu"
                )
            B, N = padding_mask.shape
            attn_bias = torch.zeros(
                B, self.num_heads, N, N, device=padding_mask.device
            )  # [B, H, N, N]

        # Apply padding mask (mask out padding nodes)
        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_bias = attn_bias.masked_fill(~pad_mask, float("-inf"))  # [B, H, N, N]

            pad_mask_row = padding_mask.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1]
            attn_bias = attn_bias.masked_fill(~pad_mask_row, float("-inf"))  # [B, H, N, N]

        return attn_bias
