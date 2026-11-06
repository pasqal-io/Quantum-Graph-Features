"""Positional / structural encodings for the downstream Graph Transformer benchmark.

Port of the encodings of the original internal implementation (``PEBenchmarkTransform``
and the helpers it uses), restricted to the feature sets of the paper (Tables 2-3 and the
geometry-baseline table). Arithmetic is kept identical to the original so that the
transformed tensors match it to float32 rounding:

- every feature block is z-scored per feature dimension with mean/std computed on the
  training split only (std is the unbiased torch std, plus ``eps = 1e-6``);
- ``*_stat`` blocks are reduced to [mean, std, min, max] over time *after* z-scoring;
- ``*_mean`` blocks average time steps 6..20 *before* the statistics are computed;
- concatenation order is: AFM part, FM part, RRWP, Laplacian, rings, geometry,
  FM residual, AFM residual, full AFM, full FM (as in the original ``__call__``);
- the ``rrwp_lap_ring_geom*`` configs are the strong baseline of the other GNN backbones,
  whose original transform used different ring (cycle basis, unscaled) and geometry
  (5 one-body / 3 two-body channels) encodings than the GT runs (``rrwp_lap_ring_geo*``).

Raw quantum features are read from attributes ``onebody_<src>`` [T, N] and
``twobody_<src>`` [T, N, N] attached to each graph (see :mod:`qgf.downstream.gt.data`).
Model-side layout is node-major: onebody [N, d], twobody [N, N, d].

Dropped from the original (not used by the paper): ground-state / HES / local-detuning
pulses, the XY pulse, ``hw_afm_full_stat`` and the ``maxabs`` / ``zscore_diag``
normalisations (the paper configs all use ``zscore``).
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected

EPS = 1e-6

# Timestep slice for mean-reduced PE: mean over timesteps 6..20 (inclusive)
_MEAN_SLICE_START = 6
_MEAN_SLICE_END = 21  # exclusive (Python slice convention)
# Extra timestep indices for mean_plus configs (0-indexed; "step N" = index N-1)
_PLUS_IDX = {"mean_plus_4": 3, "mean_plus_5": 4}

_LAPLACIAN_PE_K: int = 4
RING_SIZES: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
_RING_COL: dict[int, int] = {size: idx for idx, size in enumerate(RING_SIZES)}
# pulser C6 coefficient for Rydberg level 70 (rad um^6 / us): a global scale that cancels
# after normalisation; kept for faithfulness to the original features.
C6_PULSER_N70: float = 5420158.53


# ---------------------------------------------------------------------------
# Config table: config name -> ordered list of (source, kind) feature blocks.
# Quantum sources: afm, fm, hw_afm (hardware AFM), afm_res, fm_res (reconstruction
# residuals), afm_exact, fm_exact, afm_exact_nn, fm_exact_nn (exact MUTAG subset).
# Classical sources: rrwp, lap, ring, geo (GT runs), ring_cb, geom (GNN runs).
# Kinds: full, stat, mean, mean_plus_4, mean_plus_5, raw (unscaled).
# ---------------------------------------------------------------------------

_GEO = [("rrwp", "full"), ("lap", "full"), ("ring", "full"), ("geo", "full")]
_GEOM = [("rrwp", "full"), ("lap", "full"), ("ring_cb", "raw"), ("geom", "full")]


def _build_configs() -> dict[str, list[tuple[str, str]]]:
    cfgs: dict[str, list[tuple[str, str]]] = {
        "no_pe": [],
        "random": [],
        "laplacian": [("lap", "full")],
        "rrwp": [("rrwp", "full")],
        "rrwp_laplacian": [("rrwp", "full"), ("lap", "full")],
    }
    quantum = {}
    for kind, suffix in [
        ("full", ""),
        ("stat", "_stat"),
        ("mean", "_mean"),
        ("mean_plus_4", "_mean_plus_4"),
        ("mean_plus_5", "_mean_plus_5"),
    ]:
        quantum[f"af{suffix}"] = [("afm", kind)]
        quantum[f"fm{suffix}"] = [("fm", kind)]
    quantum["af_fm"] = [("afm", "full"), ("fm", "full")]
    quantum["af_stat_fm_stat"] = [("afm", "stat"), ("fm", "stat")]
    quantum["af_mean_fm_mean"] = [("afm", "mean"), ("fm", "mean")]
    quantum["hw_afm"] = [("hw_afm", "full")]
    quantum["hw_afm_stat"] = [("hw_afm", "stat")]
    quantum["hw_afm_3t"] = [("hw_afm_3t", "full")]  # 3 hardware times, ESA Table 4
    quantum["hw_afm_3t_stat"] = [("hw_afm_3t", "stat")]
    for name, parts in quantum.items():
        if name != "af_stat_fm_stat":  # only exists with RRWP in the original
            cfgs[name] = parts
        cfgs[f"rrwp_{name}"] = parts + [("rrwp", "full")]
    geo_extra = {
        "": [],
        "_fmres": [("fm_res", "full")],
        "_afmres": [("afm_res", "full")],
        "_fmres_afmres": [("fm_res", "full"), ("afm_res", "full")],
        "_fmres_stat": [("fm_res", "stat")],
        "_afmres_stat": [("afm_res", "stat")],
        "_fmres_afmres_stat": [("fm_res", "stat"), ("afm_res", "stat")],
        "_af_fm": [("afm", "full"), ("fm", "full")],
        "_af_fm_stat": [("afm", "stat"), ("fm", "stat")],
    }
    for suffix, parts in geo_extra.items():
        cfgs[f"rrwp_lap_ring_geo{suffix}"] = _GEO + parts
    # Strong classical baseline of the GNN runs (other backbones of the geometry table):
    # cycle-basis ring membership (not z-scored) and the 5 / 3 register-geometry channels;
    # quantum blocks come first, as in the original GNN transform.
    # "_exact" / "_exact_nn": exact (noise-free) AFM + FM of the register couplings / of the
    # graph Hamiltonian J_NN A (MUTAG subset "exact173", experiments/non_unit_disk).
    for prefix, parts in {
        "": [],
        "_af_fm": [("afm", "full"), ("fm", "full")],
        "_exact": [("afm_exact", "full"), ("fm_exact", "full")],
        "_exact_nn": [("afm_exact_nn", "full"), ("fm_exact_nn", "full")],
    }.items():
        cfgs[f"rrwp_lap_ring_geom{prefix}"] = parts + _GEOM
    return cfgs


CONFIGS: dict[str, list[tuple[str, str]]] = _build_configs()
QUANTUM_SOURCES = (
    "afm",
    "fm",
    "hw_afm",
    "hw_afm_3t",
    "afm_res",
    "fm_res",
    "afm_exact",
    "fm_exact",
    "afm_exact_nn",
    "fm_exact_nn",
)


def quantum_sources(config: str) -> list[str]:
    """Quantum sources a config needs attached to its graphs."""
    return [s for s, _ in CONFIGS[config] if s in QUANTUM_SOURCES]


# ---------------------------------------------------------------------------
# Classical encodings (verbatim arithmetic from the original)
# ---------------------------------------------------------------------------


def compute_rrwp(data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k-step random-walk probabilities: onebody [N, k] (diagonals), twobody [N, N, k]."""
    num_nodes = data.x.shape[0]  # N
    edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)  # [2, E']
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)  # [N, N]
    deg = adj.sum(dim=1, keepdim=True)  # [N, 1]
    deg = torch.where(deg > 0, deg, torch.ones_like(deg))  # Avoid div by zero
    transition = adj / deg  # [N, N]

    onebody_list: list[torch.Tensor] = []
    twobody_list: list[torch.Tensor] = []
    current = transition.clone()  # [N, N]
    for step in range(1, k + 1):
        onebody_list.append(torch.diag(current).unsqueeze(1))  # [N, 1]
        twobody_list.append(current.unsqueeze(2))  # [N, N, 1]
        if step < k:
            current = torch.matmul(current, transition)  # [N, N]
    return torch.cat(onebody_list, dim=1), torch.cat(twobody_list, dim=2)  # [N,k], [N,N,k]


def compute_laplacian_pe(
    edge_index: torch.Tensor, num_nodes: int, k: int = _LAPLACIAN_PE_K
) -> tuple[torch.Tensor, torch.Tensor]:
    """k smallest non-trivial eigenvectors of the normalized Laplacian and their outer products."""
    if num_nodes <= 1:
        return torch.zeros(num_nodes, k), torch.zeros(num_nodes, num_nodes, k)

    edge_index_ud = to_undirected(edge_index, num_nodes=num_nodes)  # [2, E']
    adj = to_dense_adj(edge_index_ud, max_num_nodes=num_nodes).squeeze(0)  # [N, N]
    deg = adj.sum(dim=1)  # [N]
    deg_inv_sqrt = torch.zeros_like(deg)  # [N]
    nonzero_mask = deg > 0  # [N]
    deg_inv_sqrt[nonzero_mask] = deg[nonzero_mask].pow(-0.5)  # [N]
    deg_inv_sqrt_diag = torch.diag(deg_inv_sqrt)  # [N, N]
    identity = torch.eye(num_nodes, dtype=adj.dtype)  # [N, N]
    laplacian = identity - deg_inv_sqrt_diag @ adj @ deg_inv_sqrt_diag  # [N, N]
    _eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)  # [N], [N, N]

    available = min(k, num_nodes - 1)
    phis = eigenvectors[:, 1 : 1 + available].clone()  # [N, available]
    # Sign convention per eigenvector: make first nonzero component positive
    for j in range(available):
        col = phis[:, j]  # [N]
        nonzero_idx = torch.nonzero(col.abs() > 1e-8, as_tuple=False)  # [?, 1]
        if nonzero_idx.numel() > 0 and col[nonzero_idx[0].item()] < 0:
            phis[:, j] = -col
    if available < k:
        pad = torch.zeros(num_nodes, k - available, dtype=phis.dtype)  # [N, k-available]
        phis = torch.cat([phis, pad], dim=1)  # [N, k]
    return phis, phis.unsqueeze(1) * phis.unsqueeze(0)  # [N, k], [N, N, k]


def compute_ring_pe(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary co-membership in minimum-cycle-basis rings of size 3..8: [N, 6], [N, N, 6]."""
    num_nodes = data.x.shape[0]  # N
    twobody = torch.zeros(num_nodes, num_nodes, len(RING_SIZES))  # [N, N, 6]
    if num_nodes <= 2:
        return torch.zeros(num_nodes, len(RING_SIZES)), twobody

    edge_index_ud = to_undirected(data.edge_index, num_nodes=num_nodes)  # [2, E']
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(edge_index_ud[0].tolist(), edge_index_ud[1].tolist()))
    for cycle in nx.minimum_cycle_basis(graph):
        col = _RING_COL.get(len(cycle))
        if col is None:
            continue
        for a in cycle:
            for b in cycle:
                twobody[a, b, col] = 1.0  # co-membership incl. diagonal a==b
    idx = torch.arange(num_nodes)  # [N]
    return twobody[idx, idx], twobody  # [N, 6], [N, N, 6]


def compute_geometry_pe(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Register geometry: twobody (C6/r^6, r) [N, N, 2]; onebody = row sums [N, 2].

    Reads ``data.pos`` (atom positions in um; the original read ``embedded_position``,
    same values).
    """
    pos = data.pos.float()  # [N, 2]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 2]
    dist = torch.linalg.norm(diff, dim=-1)  # [N, N]
    positive = dist > 0  # [N, N]
    interaction = torch.zeros_like(dist)  # [N, N]
    interaction[positive] = C6_PULSER_N70 / dist[positive].pow(6)  # [N, N]
    distance = torch.where(positive, dist, torch.zeros_like(dist))  # [N, N]
    twobody = torch.stack([interaction, distance], dim=-1)  # [N, N, 2]
    onebody = torch.stack([interaction.sum(dim=1), distance.sum(dim=1)], dim=-1)  # [N, 2]
    return onebody, twobody


def compute_ring_cycle_basis_pe(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring membership of the GNN runs: ``networkx.cycle_basis`` cycles of size 3..8.

    Onebody [N, 6]: the node lies on a basis cycle of that size; twobody [N, N, 6]: both
    endpoints lie on the same such cycle (incl. the diagonal). Binary, used unscaled.
    """
    num_nodes = data.x.shape[0]  # N
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(data.edge_index.cpu().T.numpy())
    ob = torch.zeros(num_nodes, len(RING_SIZES))  # [N, 6]
    tb = torch.zeros(num_nodes, num_nodes, len(RING_SIZES))  # [N, N, 6]
    for cycle in nx.cycle_basis(graph):
        col = _RING_COL.get(len(cycle))
        if col is None:
            continue
        idx = torch.tensor(cycle, dtype=torch.long)  # [c]
        ob[idx, col] = 1.0
        tb[idx.unsqueeze(1), idx.unsqueeze(0), col] = 1.0
    return ob, tb


def compute_register_geometry_pe(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Register geometry of the GNN runs (``r`` in um, all reductions over ``j != i``).

    Twobody [N, N, 3]: ``[r, log r, r^-6]`` with a zero diagonal; onebody [N, 5]:
    ``[sum r^-6, max r^-6, min r, mean r, distance to the centroid]``.
    """
    eps = 1e-6
    pos = data.pos.float()  # [N, 2]
    num_nodes = pos.shape[0]  # N
    r = torch.cdist(pos, pos)  # [N, N]
    eye = torch.eye(num_nodes, dtype=torch.bool)  # [N, N]
    r_safe = r.masked_fill(eye, 1.0).clamp_min(eps)  # [N, N]
    inv6 = r_safe.pow(-6.0)  # [N, N]
    tb = torch.stack([r, torch.log(r_safe), inv6], dim=-1)  # [N, N, 3]
    tb[eye] = 0.0
    if num_nodes == 1:
        return torch.zeros(1, 5), tb
    inv6_off = inv6.masked_fill(eye, 0.0)  # [N, N]
    r_for_min = r.masked_fill(eye, float("inf"))  # [N, N]
    ob = torch.stack(
        [
            inv6_off.sum(dim=1),
            inv6_off.max(dim=1).values,
            r_for_min.min(dim=1).values,
            r.sum(dim=1) / (~eye).sum(dim=1),
            (pos - pos.mean(dim=0, keepdim=True)).norm(dim=1),
        ],
        dim=-1,
    )  # [N, 5]
    return ob, tb


# ---------------------------------------------------------------------------
# Time reductions
# ---------------------------------------------------------------------------


def _reduce_time(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Pre-normalisation reduction of the last (time) dim for mean / mean_plus kinds."""
    mean_feat = x[..., _MEAN_SLICE_START:_MEAN_SLICE_END].mean(dim=-1, keepdim=True)
    if kind == "mean":
        return mean_feat  # [..., 1]
    t = _PLUS_IDX[kind]
    return torch.cat([mean_feat, x[..., t : t + 1]], dim=-1)  # [..., 2]


def to_stats(x: torch.Tensor) -> torch.Tensor:
    """Reduce last dimension to 4 via [mean, std, min, max]."""
    return torch.stack(
        [x.mean(dim=-1), x.std(dim=-1), x.min(dim=-1).values, x.max(dim=-1).values], dim=-1
    )  # [..., 4]


def _zscore_stats(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    all_values = torch.cat(values, dim=0)  # [total, d]
    return all_values.mean(dim=0), all_values.std(dim=0) + EPS  # [d], [d]


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class PEBenchmarkTransform:
    """Build the node / pair encodings of one benchmark config.

    Args:
        config: key of :data:`CONFIGS`.
        train_dataset: training graphs (with the needed quantum sources attached);
            all normalisation statistics are computed on it only.
        rrwp_k: number of RRWP steps (paper: 20).
        random_dim: dimension of the random baseline features.
    """

    def __init__(
        self, config: str, train_dataset: list[Data], rrwp_k: int = 20, random_dim: int = 20
    ) -> None:
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}, got {config!r}")
        self.config = config
        self.rrwp_k = rrwp_k
        self.random_dim = random_dim
        self.parts = CONFIGS[config]
        # (source, kind) -> (ob_mean, ob_std, tb_mean, tb_std)
        self.stats: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}
        for source, kind in self.parts:
            if kind == "raw":  # used unscaled
                continue
            stat_kind = kind if kind.startswith("mean") else "full"  # stat uses full z-score
            obs, tbs = [], []
            for d in train_dataset:
                ob, tb = self._raw(d, source)  # [N, d], [N, N, d]
                if stat_kind != "full":
                    ob, tb = _reduce_time(ob, stat_kind), _reduce_time(tb, stat_kind)
                obs.append(ob)
                tbs.append(tb.reshape(-1, tb.shape[-1]))  # [N*N, d]
            self.stats[(source, kind)] = (*_zscore_stats(obs), *_zscore_stats(tbs))

    def _raw(self, data: Data, source: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-normalised node-major features of one source: [N, d], [N, N, d]."""
        if source == "rrwp":
            return compute_rrwp(data, self.rrwp_k)
        if source == "lap":
            return compute_laplacian_pe(data.edge_index, data.x.shape[0])
        if source == "ring":
            return compute_ring_pe(data)
        if source == "geo":
            return compute_geometry_pe(data)
        if source == "ring_cb":
            return compute_ring_cycle_basis_pe(data)
        if source == "geom":
            return compute_register_geometry_pe(data)
        ob = getattr(data, f"onebody_{source}")  # [T, N]
        tb = getattr(data, f"twobody_{source}")  # [T, N, N]
        return ob.T, tb.permute(1, 2, 0)  # [N, T], [N, N, T]

    def __call__(self, data: Data) -> Data:
        result = data.clone()
        for src in QUANTUM_SOURCES:  # raw per-time tensors must not reach the collate
            for attr in (f"onebody_{src}", f"twobody_{src}"):
                if attr in result:
                    delattr(result, attr)
        num_nodes = data.x.shape[0]  # N
        if self.config == "no_pe":
            result.onebody_features = None
            result.twobody_features = None
            return result
        if self.config == "random":
            result.onebody_features = torch.randn(num_nodes, self.random_dim)  # [N, D]
            result.twobody_features = torch.randn(num_nodes, num_nodes, self.random_dim)
            return result

        ob_parts: list[torch.Tensor] = []
        tb_parts: list[torch.Tensor] = []
        for source, kind in self.parts:
            ob, tb = self._raw(data, source)  # [N, d], [N, N, d]
            if kind == "raw":
                ob_parts.append(ob)
                tb_parts.append(tb)
                continue
            if kind.startswith("mean"):
                ob, tb = _reduce_time(ob, kind), _reduce_time(tb, kind)
            ob_mean, ob_std, tb_mean, tb_std = self.stats[(source, kind)]
            ob = (ob - ob_mean) / ob_std  # [N, d]
            tb = (tb - tb_mean) / tb_std  # [N, N, d]
            if kind == "stat":
                ob, tb = to_stats(ob), to_stats(tb)  # [N, 4], [N, N, 4]
            ob_parts.append(ob)
            tb_parts.append(tb)
        result.onebody_features = torch.cat(ob_parts, dim=-1)  # [N, ob_dim]
        result.twobody_features = torch.cat(tb_parts, dim=-1)  # [N, N, tb_dim]
        return result

    def __repr__(self) -> str:
        return f"PEBenchmarkTransform(config={self.config!r}, rrwp_k={self.rrwp_k})"


__all__ = [
    "CONFIGS",
    "PEBenchmarkTransform",
    "compute_geometry_pe",
    "compute_laplacian_pe",
    "compute_register_geometry_pe",
    "compute_ring_cycle_basis_pe",
    "compute_ring_pe",
    "compute_rrwp",
    "quantum_sources",
    "to_stats",
]
