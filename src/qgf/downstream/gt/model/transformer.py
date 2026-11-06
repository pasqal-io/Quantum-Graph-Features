"""Graph Transformer architecture."""

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from qgf.downstream.gt.model.attention import GatedMultiHeadAttention, MultiHeadAttention
from qgf.downstream.gt.model.config import ModelConfig
from qgf.downstream.gt.model.embeddings import (
    AbsolutePEEncoder,
    EdgeEmbedding,
    NodeEmbedding,
    RelativePEEncoder,
)
from qgf.downstream.gt.model.pooling import MeanPooling, SumPooling
from qgf.downstream.gt.model.twobody_attention import (
    DeepBiasEncoder,
    NodeEnrichment,
    TwobodyToEdgeProjector,
)

# Strategies that produce a rel_pe_bias [B,N,N,H] (fed to standard attention)
_BIAS_STRATEGIES = {"bias", "deep_bias"}

# Strategies that enrich embeddings before attention (no rel_pe_bias needed)
_EMBEDDING_STRATEGIES = {"node_enrichment", "edge_enrichment", "edge_fusion"}


class TransformerBlock(nn.Module):
    """
    Single Graph Transformer block.

    Structure:
        1. Multi-head attention with edge and PE bias
        2. Add & Norm
        3. Feed-forward network
        4. Add & Norm
    """

    def __init__(self, config: ModelConfig, edge_bias_dim: int = 0) -> None:
        super().__init__()

        # All strategies use standard attention (gated or not)
        if config.use_gated_attention:
            self.attention = GatedMultiHeadAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                bias_scale_init=config.bias_scale_init,
                edge_bias_dim=edge_bias_dim,
            )
        else:
            self.attention = MultiHeadAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                edge_bias_dim=edge_bias_dim,
            )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_embed_dense: torch.Tensor | None,
        rel_pe_bias: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Node features
            edge_embed_dense: [B, N, N, D] or None - Edge embeddings (zeros for non-bonded pairs)
            rel_pe_bias: [B, N, N, H] or None - Relative PE bias
            padding_mask: [B, N] or None - Valid node mask

        Returns:
            x: [B, N, D] - Updated node features
        """
        # Multi-head attention with residual
        attn_out = self.attention(x, edge_embed_dense, rel_pe_bias, padding_mask)  # [B, N, D]
        x = self.norm1(x + attn_out)  # [B, N, D]

        # Feed-forward with residual
        ffn_out = self.ffn(x)  # [B, N, D]
        x = self.norm2(x + ffn_out)  # [B, N, D]

        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer for molecular property prediction.

    Architecture:
        1. Embed nodes, edges, and positional encodings
        2. Apply twobody strategy (enrich embeddings or produce bias)
        3. Stack of Transformer blocks
        4. Graph pooling (mean)
        5. Output MLP for regression
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        node_edim = config.effective_node_embed_dim

        # --- Node embeddings (potentially smaller than D) ---
        self.node_embed = NodeEmbedding(config.num_atom_types, node_edim)

        # --- Absolute PE encoder (always projects to D) ---
        self.abs_pe_encoder = (
            AbsolutePEEncoder(config.abs_pe_dim, D) if config.abs_pe_dim > 0 else None
        )

        # --- Node fusion: project (node_embed || pe_encoded) or node_embed alone to D ---
        # When node_edim != D, we need a projection to reach hidden_dim
        self._node_uses_small_embed = node_edim != D
        self.concat_proj: nn.Linear | None = None
        self.node_proj: nn.Linear | None = None

        if self._node_uses_small_embed:
            if config.abs_pe_dim > 0:
                # cat(node_embed[node_edim], pe_encoded[D]) -> Linear -> [D]
                self.concat_proj = nn.Linear(node_edim + D, D)
            else:
                # Just project small embed to D
                self.node_proj = nn.Linear(node_edim, D)
        elif config.pe_fusion_mode == "concat" and config.abs_pe_dim > 0:
            # concat fusion mode: cat(node_embed[D], pe_encoded[D]) -> Linear -> [D]
            self.concat_proj = nn.Linear(2 * D, D)

        # --- Edge embeddings (always at hidden_dim D) ---
        self.edge_embed = EdgeEmbedding(config.num_bond_types, D)

        # --- Twobody strategy modules ---
        self._twobody_strategy = config.twobody_strategy
        self.rel_pe_encoder: RelativePEEncoder | None = None
        self.twobody_bias_encoder: DeepBiasEncoder | None = None
        self.twobody_node_enrichment: NodeEnrichment | None = None
        self.twobody_edge_projector: TwobodyToEdgeProjector | None = None

        if config.rel_pe_dim > 0:
            self._init_twobody_modules(config)

        # --- Determine edge_bias_dim for attention blocks ---
        # Always D: attention's edge_bias_proj handles D → H projection.
        edge_bias_dim = D

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, edge_bias_dim=edge_bias_dim)
                for _ in range(config.num_layers)
            ]
        )

        # Graph pooling
        self.pooling = SumPooling() if config.pooling == "sum" else MeanPooling()

        # Output head
        output_dim = config.num_classes if config.task_type == "classification" else 1
        self.output_head = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(D, output_dim),
        )

    def _init_twobody_modules(self, config: ModelConfig) -> None:
        """Initialize twobody-specific modules based on strategy."""
        strategy = config.twobody_strategy

        if strategy == "bias":
            # Baseline: simple RelativePEEncoder → [B,N,N,H]
            self.rel_pe_encoder = RelativePEEncoder(config.rel_pe_dim, config.num_heads)

        elif strategy == "deep_bias":
            # Deeper MLP → [B,N,N,H], additive to edge bias
            self.twobody_bias_encoder = DeepBiasEncoder(
                config.rel_pe_dim, config.hidden_dim, config.num_heads
            )

        elif strategy == "node_enrichment":
            # Mean+max pool → enrich node embeddings
            self.twobody_node_enrichment = NodeEnrichment(config.rel_pe_dim, config.hidden_dim)

        elif strategy in ("edge_enrichment", "edge_fusion"):
            # Project twobody → D, add to edge embeddings
            self.twobody_edge_projector = TwobodyToEdgeProjector(
                config.rel_pe_dim, config.hidden_dim
            )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data: PyG Batch object with:
                - x: [total_N] - Node features
                - edge_index: [2, total_E] - Edge connectivity
                - edge_attr: [total_E] - Edge features
                - batch: [total_N] - Batch assignment
                - onebody_features: [total_N, d_abs] - Absolute PE (if abs_pe_dim > 0)
                - twobody_features: List[Tensor[N_i, N_i, d_rel]] - Relative PE (if rel_pe_dim > 0)

        Returns:
            y_pred: [B] - Predicted property values
        """
        batch_size = data.batch.max().item() + 1
        device = data.x.device

        # Ensure x is 1D
        x = data.x.squeeze(-1) if len(data.x.shape) == 2 else data.x  # [total_N]

        # --- Node path ---
        h_node = self._build_node_features(x, data, device)  # [total_N, D]

        # --- Edge path ---
        edge_attr = (
            data.edge_attr.squeeze(-1) if len(data.edge_attr.shape) == 2 else data.edge_attr
        )  # [total_E]
        h_edge = self.edge_embed(edge_attr)  # [total_E, D]

        # Convert to dense batch format for attention
        h_node_dense, padding_mask = to_dense_batch(h_node, data.batch)  # [B, N, D], [B, N]

        # Convert edge embeddings to dense format
        h_edge_dense = self._edges_to_dense(
            data.edge_index, h_edge, data.batch, padding_mask
        )  # [B, N, N, D]

        # --- Twobody / fusion path ---
        rel_pe_bias, h_edge_for_attn = self._build_edge_and_twobody(
            data, batch_size, padding_mask, h_node_dense, h_edge_dense
        )

        # Apply transformer blocks
        for block in self.blocks:
            h_node_dense = block(
                h_node_dense, h_edge_for_attn, rel_pe_bias, padding_mask
            )  # [B, N, D]

        # Pool to graph-level representation
        h_node_flat = h_node_dense[padding_mask]  # [total_N, D]
        h_graph = self.pooling(h_node_flat, data.batch)  # [B, D]

        # Predict property
        if self.config.task_type == "classification":
            y_pred = self.output_head(h_graph)  # [B, num_classes]
        else:
            y_pred = self.output_head(h_graph).squeeze(-1)  # [B, 1] -> [B]

        return y_pred

    def _build_node_features(
        self, x: torch.Tensor, data: Batch, device: torch.device
    ) -> torch.Tensor:
        """Build node features with optional small-embedding fusion.

        Args:
            x: [total_N] - Categorical node features
            data: Batch with onebody_features
            device: Target device

        Returns:
            h_node: [total_N, D] - Node features at hidden_dim
        """
        h_node = self.node_embed(x)  # [total_N, node_edim]

        if self.abs_pe_encoder is not None:
            abs_pe = torch.cat(
                [pe.to(device) for pe in data.onebody_features], dim=0
            )  # [total_N, d_abs]
            pe_encoded = self.abs_pe_encoder(abs_pe)  # [total_N, D]

            if self.concat_proj is not None:
                # Concat mode: cat(node_embed, pe_encoded) -> Linear -> [D]
                h_node = self.concat_proj(torch.cat([h_node, pe_encoded], dim=-1))  # [total_N, D]
            else:
                # Add mode (only when node_edim == D)
                h_node = h_node + pe_encoded  # [total_N, D]
        elif self.node_proj is not None:
            # No PE but small embed -> project to D
            h_node = self.node_proj(h_node)  # [total_N, D]

        return h_node

    def _build_edge_and_twobody(
        self,
        data: Batch,
        batch_size: int,
        padding_mask: torch.Tensor,
        h_node_dense: torch.Tensor,
        h_edge_dense: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply twobody strategy and return (rel_pe_bias, edge_embed_for_attention).

        Args:
            data: Batch with twobody_features
            batch_size: Number of graphs
            padding_mask: [B, N]
            h_node_dense: [B, N, D]
            h_edge_dense: [B, N, N, D]

        Returns:
            rel_pe_bias: [B, N, N, H] or None (for bias strategies)
            h_edge_for_attn: [B, N, N, D] — edge embeddings passed to attention
        """
        rel_pe_bias = self._apply_twobody_strategy(
            data, batch_size, padding_mask, h_node_dense, h_edge_dense
        )  # [B, N, N, H] or None

        return rel_pe_bias, h_edge_dense

    def _apply_twobody_strategy(
        self,
        data: Batch,
        batch_size: int,
        padding_mask: torch.Tensor,
        h_node_dense: torch.Tensor,
        h_edge_dense: torch.Tensor,
    ) -> torch.Tensor | None:
        """Apply twobody strategy: produce bias or enrich embeddings in-place.

        Args:
            data: Batch with twobody_features
            batch_size: Number of graphs in batch
            padding_mask: [B, N]
            h_node_dense: [B, N, D] — modified in-place for node_enrichment
            h_edge_dense: [B, N, N, D] — modified in-place for edge strategies

        Returns:
            rel_pe_bias: [B, N, N, H] or None (only for bias strategies)
        """
        strategy = self._twobody_strategy

        # Baseline bias: use RelativePEEncoder
        if strategy == "bias":
            return self._encode_relative_pe(data, batch_size, padding_mask)

        # All other strategies need raw batched twobody
        twobody_raw = self._batch_twobody_raw(data, batch_size, padding_mask)
        if twobody_raw is None:
            return None

        if strategy == "deep_bias":
            assert self.twobody_bias_encoder is not None
            return self.twobody_bias_encoder(twobody_raw)  # [B, N, N, H]

        if strategy == "node_enrichment":
            assert self.twobody_node_enrichment is not None
            enrichment = self.twobody_node_enrichment(twobody_raw, padding_mask)  # [B, N, D]
            h_node_dense += enrichment  # in-place enrichment
            return None

        if strategy in ("edge_enrichment", "edge_fusion"):
            assert self.twobody_edge_projector is not None
            edge_enrichment = self.twobody_edge_projector(twobody_raw)  # [B, N, N, D]
            h_edge_dense += edge_enrichment  # in-place enrichment
            return None

        return None

    def _edges_to_dense(
        self,
        edge_index: torch.Tensor,
        edge_embed: torch.Tensor,
        batch: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert sparse edge embeddings to dense format.

        Args:
            edge_index: [2, E] - Edge connectivity (global indices)
            edge_embed: [E, D] - Edge embeddings
            batch: [total_N] - Batch assignment for nodes
            padding_mask: [B, max_N] - Valid node mask

        Returns:
            edge_embed_dense: [B, max_N, max_N, D] - Dense edge embeddings
        """
        B, max_N = padding_mask.shape
        D = edge_embed.shape[1]
        device = edge_embed.device

        # Initialize dense edge embedding tensor (zeros for no-edge)
        edge_embed_dense = torch.zeros(B, max_N, max_N, D, device=device)  # [B, N, N, D]

        # Compute batch assignment for edges (which graph each edge belongs to)
        src_batch = batch[edge_index[0]]  # [E]
        dst_batch = batch[edge_index[1]]  # [E]

        # Edges should be within same graph
        assert (src_batch == dst_batch).all(), "Edges must be within same graph"
        edge_batch = src_batch  # [E]

        # Compute node offsets for each graph in batch
        # node_offset[b] = starting index of graph b in the flattened node list
        node_offset = torch.zeros(B + 1, dtype=torch.long, device=device)  # [B+1]
        for b in range(B):
            node_offset[b + 1] = node_offset[b] + padding_mask[b].sum().item()

        # Convert global edge indices to local (per-graph) indices and fill dense tensor
        for b in range(B):
            # Get edges belonging to graph b
            edge_mask = edge_batch == b  # [E]
            if not edge_mask.any():
                continue

            # Local edge indices (relative to graph b)
            local_src = edge_index[0, edge_mask] - node_offset[b]  # [E_b]
            local_dst = edge_index[1, edge_mask] - node_offset[b]  # [E_b]

            # Get edge embeddings for this graph
            edge_embed_b = edge_embed[edge_mask]  # [E_b, D]

            # Fill dense tensor
            edge_embed_dense[b, local_src, local_dst] = edge_embed_b  # [E_b, D]

        return edge_embed_dense

    def _encode_relative_pe(
        self, data: Batch, batch_size: int, padding_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Encode relative PE to attention bias.

        Args:
            data: Batch with twobody_features as List[Tensor[N_i, N_i, d_rel]]
            batch_size: Number of graphs in batch
            padding_mask: [B, N] - Valid node mask

        Returns:
            rel_pe_bias: [B, N, N, H] or None
        """
        if self.rel_pe_encoder is None:
            return None

        # twobody_features is a list of [N_i, N_i, d_rel] tensors
        # Batch them into [B, max_N, max_N, d_rel]
        max_nodes = padding_mask.shape[1]
        device = padding_mask.device

        batched_twobody = torch.zeros(
            batch_size, max_nodes, max_nodes, self.config.rel_pe_dim, device=device
        )  # [B, max_N, max_N, d_rel]

        for i, pe in enumerate(data.twobody_features):
            n = pe.shape[0]
            batched_twobody[i, :n, :n, :] = pe.to(device)  # [N_i, N_i, d_rel]

        return self.rel_pe_encoder(batched_twobody)  # [B, max_N, max_N, H]

    def _batch_twobody_raw(
        self, data: Batch, batch_size: int, padding_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Batch raw twobody features into dense format (without encoding).

        Args:
            data: Batch with twobody_features as List[Tensor[N_i, N_i, d_rel]]
            batch_size: Number of graphs in batch
            padding_mask: [B, N] - Valid node mask

        Returns:
            batched_twobody: [B, N, N, d_rel] or None
        """
        if self.config.rel_pe_dim == 0:
            return None
        if not hasattr(data, "twobody_features") or data.twobody_features is None:
            return None

        max_nodes = padding_mask.shape[1]
        device = padding_mask.device

        batched_twobody = torch.zeros(
            batch_size, max_nodes, max_nodes, self.config.rel_pe_dim, device=device
        )  # [B, max_N, max_N, d_rel]

        for i, pe in enumerate(data.twobody_features):
            n = pe.shape[0]
            batched_twobody[i, :n, :n, :] = pe.to(device)  # [N_i, N_i, d_rel]

        return batched_twobody  # [B, max_N, max_N, d_rel]
