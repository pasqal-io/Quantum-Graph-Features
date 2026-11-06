"""Model configuration dataclass."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Graph Transformer model.

    Fixed hyperparameters for fair PE comparison:
    - hidden_dim: Node embedding dimension
    - num_heads: Number of attention heads
    - num_layers: Number of transformer blocks
    - ffn_dim: FFN hidden dimension (2x hidden_dim)
    - dropout: Dropout rate

    Gated attention (for preventing bias explosion):
    - use_gated_attention: Use tanh-gated biases instead of raw biases
    - bias_scale_init: Initial value for learnable bias scale (λ)

    Variable PE dimensions:
    - abs_pe_dim: Absolute PE dimension (0 = disabled)
    - rel_pe_dim: Relative PE dimension (0 = disabled)

    PE fusion mode:
    - pe_fusion_mode: How to combine node embeddings with absolute PE
      - "add": h_node = node_embed + pe_encoded (default)
      - "concat": h_node = Linear(concat(node_embed, pe_encoded))

    Twobody injection strategy (all SDPA-compatible):
    - twobody_strategy: How to inject twobody features
      - "bias": Compress to H scalars as additive attention bias (default, existing)
      - "edge_fusion": Fuse twobody projection with edge embeddings before bias projection
      - "deep_bias": Deeper MLP for twobody → H bias, additive to edge bias
      - "node_enrichment": Mean+max pool twobody per node, enrich node embeddings
      - "edge_enrichment": Add twobody projection to edge embeddings

    Flexible embedding dimensions:
    - node_embed_dim: Node embedding size before projection (0 = use hidden_dim)

    Dataset parameters:
    - num_atom_types: ZINC atom vocabulary size
    - num_bond_types: ZINC bond vocabulary size
    """

    # Fixed architecture hyperparameters
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1

    # Gated attention (prevents bias explosion)
    use_gated_attention: bool = False
    bias_scale_init: float = 1.0

    # PE dimensions (0 = disabled)
    abs_pe_dim: int = 0
    rel_pe_dim: int = 0

    # PE fusion mode: "add" (default) or "concat"
    # - "add": h_node = node_embed + pe_encoded
    # - "concat": h_node = Linear(concat(node_embed, pe_encoded))
    pe_fusion_mode: str = "add"

    # Flexible node embedding dimension (0 = use hidden_dim for baseline comparison)
    node_embed_dim: int = 0

    # Twobody injection strategy
    twobody_strategy: str = "bias"

    # Pooling mode: "mean" (default) or "sum"
    pooling: str = "mean"

    # Task configuration
    task_type: str = "regression"  # "regression" or "classification"
    num_classes: int = 1  # Number of output classes (only used for classification)

    # Dataset parameters
    num_atom_types: int = 28
    num_bond_types: int = 4

    VALID_TWOBODY_STRATEGIES = (
        "bias",
        "edge_fusion",
        "deep_bias",
        "node_enrichment",
        "edge_enrichment",
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.pe_fusion_mode not in ("add", "concat"):
            raise ValueError(
                f"pe_fusion_mode must be 'add' or 'concat', got '{self.pe_fusion_mode}'"
            )
        if self.twobody_strategy not in self.VALID_TWOBODY_STRATEGIES:
            raise ValueError(
                f"twobody_strategy must be one of {self.VALID_TWOBODY_STRATEGIES}, "
                f"got '{self.twobody_strategy}'"
            )
        if self.pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got '{self.pooling}'")
        if self.task_type not in ("regression", "classification"):
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got '{self.task_type}'"
            )
        if self.task_type == "classification" and self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2 for classification, got {self.num_classes}")

    @property
    def head_dim(self) -> int:
        """Compute head dimension (hidden_dim // num_heads)."""
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        return self.hidden_dim // self.num_heads

    @property
    def effective_node_embed_dim(self) -> int:
        """Node embedding dimension (falls back to hidden_dim when 0)."""
        return self.node_embed_dim if self.node_embed_dim > 0 else self.hidden_dim
