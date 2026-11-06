"""Edge Set Attention layers (port of the ESA model used in the paper, Table 4).

Port of the internal fork of ESA (Buterez et al., "An end-to-end attention-based approach
for learning on graphs", Nat. Commun. 2025), restricted to what the MUTAG runs used: edge
attention, ``post`` normalisation, LayerNorm, gated MLPs, no dropout.

Differences from the original, which did not change the model:

- attention: xformers' memory-efficient kernel when :func:`set_attention_backend` selects
  ``"xformers"`` (as the paper runs; needs xformers), otherwise
  ``torch.nn.functional.scaled_dot_product_attention`` with the same additive mask (0
  between adjacent edges, -99999 elsewhere). The original fork's own torch path inverted
  that mask (edges attended to non-adjacent edges); it was not used for the paper runs and
  is not reproduced;
- SwiGLU: on CUDA the fused kernel of flash-attn (``flash_attn.ops.activations.swiglu``,
  forward and backward computed in float32 and rounded once) is compiled from the same
  source with ``torch.cuda.jiterator``, so flash-attn is not needed; on CPU
  ``silu(gate) * y``;
- the Admin residual ``x * omega + f(x)`` (Liu et al., EMNLP 2020; ``admin_torch.as_module``
  with its default ``O(log n)`` output change) is implemented here (:func:`admin_residual`);
- masks are built on the device of the inputs (the original hard-coded ``cuda:0``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_normal_

MASK_FILL = -99999.0  # additive bias between non-adjacent edges (as in the original)
ATTENTION_BACKENDS = ("sdpa", "xformers")
_ATTENTION = {"backend": "sdpa"}


def set_attention_backend(name: str) -> None:
    """``"xformers"`` (memory-efficient kernel of the paper runs) or ``"sdpa"`` (PyTorch)."""
    if name not in ATTENTION_BACKENDS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}, got {name!r}")
    if name == "xformers":
        import xformers.ops  # noqa: F401  (fail early if missing)
    _ATTENTION["backend"] = name


# ---------------------------------------------------------------------------
# SwiGLU (flash-attn's fused kernel, BSD-3-Clause, Tri Dao)
# ---------------------------------------------------------------------------

_SWIGLU_FWD = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
_SWIGLU_BWD = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
_JIT: dict[str, object] = {}


def _jit(name: str):
    if name not in _JIT:
        jit = torch.cuda.jiterator
        _JIT["fwd"] = jit._create_jit_fn(_SWIGLU_FWD)
        _JIT["bwd"] = jit._create_multi_output_jit_fn(_SWIGLU_BWD, num_outputs=2)
    return _JIT[name]


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _jit("fwd")(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return _jit("bwd")(x, y, dout)


def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``silu(x) * y``; on CUDA flash-attn's fused kernel (single rounding)."""
    if x.is_cuda:
        return _SwiGLU.apply(x, y)
    return F.silu(x) * y


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class OmegaResidual(nn.Module):
    """``x * omega + f_x`` with a fixed (buffer) rescaling of the shortcut."""

    def __init__(self, omega: float) -> None:
        super().__init__()
        self.register_buffer("omega", torch.tensor([omega], dtype=torch.float32))

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        return x * self.omega + f_x


def admin_residual(num_res_layers: int) -> OmegaResidual:
    """Admin shortcut for ``num_res_layers`` residual layers: ``omega = sqrt((n + 1) /
    ln(n + 1) - 1)`` (``admin_torch.as_module(n)``)."""
    n = num_res_layers
    return OmegaResidual(((n + 1) / math.log(n + 1) - 1) ** 0.5)


class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        inter_dim: int,
        out_dim: int,
        dropout_p: float = 0.0,
        num_layers: int = 2,
        use_ln: bool = False,
    ) -> None:
        super().__init__()
        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
            return
        layers: list[nn.Module] = []
        for i in range(num_layers):
            last = i == num_layers - 1
            layers.append(
                nn.Linear(in_dim if i == 0 else inter_dim, out_dim if last else inter_dim)
            )
            if not last:
                if use_ln:
                    layers.append(nn.LayerNorm(inter_dim))
                layers.append(nn.Mish())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GatedMLPSingle(nn.Module):
    """``fc2(silu(gate) * y)`` with ``[y, gate] = fc1(x)`` (optional LayerNorm on fc1)."""

    def __init__(self, in_dim: int, inter_dim: int, out_dim: int, use_ln: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(self.fc1(x)) if self.use_ln else self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        return self.fc2(swiglu(gate, y))


class GatedMLPMulti(nn.Module):
    def __init__(
        self, in_dim: int, inter_dim: int, out_dim: int, num_layers: int = 2, use_ln: bool = False
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else inter_dim
            d_out = out_dim if i == num_layers - 1 else inter_dim
            layers += [GatedMLPSingle(d_in, inter_dim, d_out, use_ln=use_ln), nn.Mish()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def layer_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=1e-8)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MAB(nn.Module):
    """Multi-head attention block: ``out + mish(fc_o(out))``."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int) -> None:
        super().__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_v, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_v, bias=True)
        self.fc_v = nn.Linear(dim_k, dim_v, bias=True)
        self.fc_o = nn.Linear(dim_v, dim_v, bias=True)
        for fc in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            xavier_normal_(fc.weight)

    def forward(self, q: torch.Tensor, k: torch.Tensor, bias: torch.Tensor | None = None):
        b = q.size(0)
        head_dim = self.dim_v // self.num_heads
        q_, v_, k_ = self.fc_q(q), self.fc_v(k), self.fc_k(k)
        q_, k_, v_ = (t.view(b, -1, self.num_heads, head_dim) for t in (q_, k_, v_))  # [B, L, H, d]
        if bias is not None:
            bias = bias.to(q_.dtype).expand(-1, self.num_heads, -1, -1)  # [B, H, L, L]
        if _ATTENTION["backend"] == "xformers":
            from xformers.ops import memory_efficient_attention

            out = memory_efficient_attention(q_, k_, v_, attn_bias=bias, p=0)
            out = out.reshape(b, -1, self.dim_v)  # [B, L, H * d]
        else:
            q_, k_, v_ = (t.transpose(1, 2) for t in (q_, k_, v_))  # [B, H, L, d]
            out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=bias)
            out = out.transpose(1, 2).reshape(b, -1, self.dim_v)  # [B, L, H * d]
        return out + F.mish(self.fc_o(out))


class SAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int) -> None:
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, bias)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x)


def edge_adjacency_bias(
    edge_index: torch.Tensor, batch_mapping: torch.Tensor, batch_size: int, max_items: int
) -> torch.Tensor:
    """``[B, 1, L, L]`` additive bias: 0 between two edges of the same graph that share a
    node (excluding an edge with itself), ``MASK_FILL`` elsewhere (incl. padding)."""
    device = edge_index.device
    src, dst = edge_index
    adjacent = (
        (src[:, None] == src[None, :])
        | (dst[:, None] == dst[None, :])
        | (src[:, None] == dst[None, :])
        | (dst[:, None] == src[None, :])
    )
    adjacent.fill_diagonal_(False)
    edge_graph = batch_mapping[src]  # [E]
    # position of each edge inside its graph (edges are stored graph by graph)
    counts = torch.bincount(edge_graph, minlength=batch_size)
    starts = torch.cumsum(counts, 0) - counts
    local = torch.arange(edge_graph.numel(), device=device) - starts[edge_graph]
    i, j = adjacent.nonzero(as_tuple=True)
    bias = torch.full((batch_size, max_items, max_items), MASK_FILL, device=device)
    bias[edge_graph[i], local[i], local[j]] = 0.0
    return bias.unsqueeze(1)


# ---------------------------------------------------------------------------
# Encoder / decoder blocks
# ---------------------------------------------------------------------------


class SABComplete(nn.Module):
    """(Masked) self-attention + gated MLP with Admin residuals and post LayerNorm.

    ``masked`` = attention restricted to adjacent edges (the "M" layers)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        masked: bool,
        mlp_hidden_size: int,
        num_mlp_layers: int,
        use_mlp_ln: bool,
        num_res_layers: int,
    ) -> None:
        super().__init__()
        self.masked = masked
        self.residual_attn = admin_residual(num_res_layers)
        self.residual_mlp = admin_residual(num_res_layers)
        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)
        self.sab = SAB(dim_in, dim_out, num_heads)
        self.norm = layer_norm(dim_out)
        self.mlp = GatedMLPMulti(
            dim_out, mlp_hidden_size, dim_out, num_layers=num_mlp_layers, use_ln=use_mlp_ln
        )
        self.norm_mlp = layer_norm(dim_out)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        out_attn = self.sab(x, bias if self.masked else None)
        if out_attn.shape[-1] != x.shape[-1]:
            x = self.proj_1(x)
        out = self.norm(self.residual_attn(x, out_attn))
        out_mlp = self.mlp(out)
        if out.shape[-1] == out_mlp.shape[-1]:
            out = self.residual_mlp(out, out_mlp)
        return self.norm_mlp(out)


class PMAComplete(nn.Module):
    """Pooling by multi-head attention onto ``num_outputs`` seeds + gated MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_outputs: int,
        mlp_hidden_size: int,
        num_mlp_layers: int,
        use_mlp_ln: bool,
        num_res_layers: int,
    ) -> None:
        super().__init__()
        self.residual_attn = admin_residual(num_res_layers)
        self.residual_mlp = admin_residual(num_res_layers)
        self.pma = PMA(dim, num_heads, num_outputs)
        self.norm = layer_norm(dim)
        self.mlp = GatedMLPMulti(
            dim, mlp_hidden_size, dim, num_layers=num_mlp_layers, use_ln=use_mlp_ln
        )
        self.norm_mlp = layer_norm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_attn = self.pma(x)
        if out_attn.shape[-2] == x.shape[-2]:
            out = self.norm(self.residual_attn(x, out_attn))
        else:
            out = out_attn
        out_mlp = self.mlp(out)
        if out.shape[-2] == out_mlp.shape[-2]:
            out = self.residual_mlp(out, out_mlp)
        return self.norm_mlp(out)


class ESA(nn.Module):
    """Edge-set encoder ("S" self-attention, "M" masked attention) -> PMA -> "S" decoder.

    ``layer_types`` e.g. ``["S", "M", "M", "M", "M", "M", "S", "P"]``; graph embedding =
    ``mish(Linear(mean over the PMA seeds))``.
    """

    def __init__(
        self,
        num_outputs: int,
        dim_output: int,
        dim_hidden: list[int],
        num_heads: list[int],
        layer_types: list[str],
        mlp_hidden_size: int,
        num_mlp_layers: int,
        use_mlp_ln: bool,
    ) -> None:
        super().__init__()
        assert len(layer_types) == len(dim_hidden) == len(num_heads)
        num_res_layers = len(dim_hidden) * 2
        mlp = dict(
            mlp_hidden_size=mlp_hidden_size,
            num_mlp_layers=num_mlp_layers,
            use_mlp_ln=use_mlp_ln,
            num_res_layers=num_res_layers,
        )
        has_pma = "P" in layer_types
        encoder: list[nn.Module] = []
        decoder: list[nn.Module] = []
        tracker, pma_seen = 0, False
        for lt in layer_types:
            d_in, heads = dim_hidden[tracker], num_heads[tracker]
            d_out = dim_hidden[tracker + 1] if (lt != "P" and has_pma) else d_in
            if lt in ("S", "M") and not pma_seen:
                encoder.append(SABComplete(d_in, d_out, heads, masked=lt == "M", **mlp))
            elif lt == "P":
                pma_seen = True
                decoder.append(PMAComplete(d_in, heads, num_outputs, **mlp))
            elif lt == "S":
                decoder.append(SABComplete(d_in, d_out, heads, masked=False, **mlp))
            if lt != "P":
                tracker += 1
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.decoder_linear = nn.Linear(dim_hidden[-1], dim_output, bias=True)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        enc = x
        for layer in self.encoder:
            enc = layer(enc, bias)
        out = enc + x
        for i, layer in enumerate(self.decoder):
            out = layer(out) if i == 0 else layer(out, None)
        return F.mish(self.decoder_linear(out.mean(dim=1)))  # [B, dim_output]
