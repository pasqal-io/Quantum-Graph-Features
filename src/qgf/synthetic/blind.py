"""RRWP-K-blind groups of ladders (18-26 atoms).

Every open ladder with ``m`` rungs (all crossing subsets) is enumerated and its pooled
RRWP-K signature computed: the sorted entries of ``M^1 .. M^K`` with ``M = D^-1 A``, i.e.
what any permutation-invariant RRWP-K readout can see. A **blind group** is a set of
ladders sharing one signature that contains both a pinned and a frustrated graph: inside
it no RRWP-K model can beat chance. A group-balanced subset (as many pinned as frustrated
graphs per group) makes the whole dataset so.

Every K-blind graph is also (K-1)-blind, so the K = 2 pool contains the K = 3 pool.
"""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

from qgf.synthetic.ladder import gs_correlations_dp, ladder


def rrwp_signatures(G: nx.Graph, ks) -> dict[int, tuple]:
    """``{K: hashable pooled RRWP-K signature}`` (entries rounded to 1e-10)."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n))
    M = A / A.sum(1)[:, None]
    P = M.copy()
    parts, out = [], {}
    for k in range(1, max(ks) + 1):
        parts.append(np.round(np.sort(P.ravel()), 10))
        if k in ks:
            out[k] = (n, np.concatenate(parts).tobytes())
        P = P @ M
    return out


def blind_pool(m_lo: int, m_hi: int, ks) -> tuple[list[dict], dict[int, np.ndarray]]:
    """All open ladders with ``m_lo <= m <= m_hi`` rungs and at least one crossing.

    Returns ``(records, group ids)``: one record ``{"m", "cross", "y", "degeneracy"}`` per
    ladder (``y = 1`` pinned) and, per ``K`` in ``ks``, the blind-group id of each record
    (``-1`` = not in a blind group).
    """
    recs = []
    groups = {k: defaultdict(list) for k in ks}
    for m in range(m_lo, m_hi + 1):
        for r in range(1, m):
            for cr in combinations(range(m - 1), r):
                G = ladder(m, cr)
                _, _, count = gs_correlations_dp(G)
                i = len(recs)
                recs.append({"m": m, "cross": cr, "y": int(count == 1), "degeneracy": count})
                for k, s in rrwp_signatures(G, ks).items():
                    groups[k][s].append(i)
    gid = {k: np.full(len(recs), -1) for k in ks}
    for k in ks:
        g = 0
        for idx in groups[k].values():
            if len({recs[i]["y"] for i in idx}) == 2:
                gid[k][idx] = g
                g += 1
    return recs, gid


def balanced_subset(y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Sorted indices with equal numbers of pinned and frustrated graphs in every group
    (``groups < 0`` = excluded)."""
    rng = np.random.default_rng(seed)
    keep = []
    for grp in np.unique(groups[groups >= 0]):
        idx = np.nonzero(groups == grp)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        k = min(len(pos), len(neg))
        keep += list(rng.choice(pos, k, replace=False))
        keep += list(rng.choice(neg, k, replace=False))
    return np.sort(np.array(keep))
