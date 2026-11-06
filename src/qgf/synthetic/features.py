"""Permutation-invariant graph-level features of the ladder benchmark.

Every family is pooled over nodes, so a graph and any relabelling of it get the same
vector: one-body blocks ``[n, d]`` -> per-column quantiles (0, 5, 10, 25, 50, 75, 90, 95,
100 %) + mean; two-body blocks ``[n, n, d]`` -> the same statistics over off-diagonal
pairs and over edges. Values are rounded to 1e-10 so that bit-identical graphs (the blind
groups) stay bit-identical.

Families
    n            node count
    rrwp<K>      RRWP ``M^1..M^K`` (diagonal + two-body); K = 2, 3 are the blind horizons,
                 K = 20 is the paper encoding
    geom         register geometry (r, log r, r^-6; per-node coupling sums, nearest
                 distance, mean distance, distance to the centre)
    lap          Laplacian and normalised-Laplacian spectra, ``|phi_2..phi_5|``
    ring         ring membership (cycle-basis cycles of size 3-8)
    afm, fm      quench features: occupation ``<n_i>(t)`` and connected correlation
                 ``C_ij(t)`` at the 20 evaluation times
    gs, gs_max   ideal ground state (uniform over the MIS): ``<n_i>``, ``C_ij``; max ``|C_ij|``
"""

import networkx as nx
import numpy as np

from qgf.synthetic.ladder import gs_correlations_dp, ladder

QUANTILES = np.array([0, 5, 10, 25, 50, 75, 90, 95, 100]) / 100
RING_SIZES = (3, 4, 5, 6, 7, 8)
RRWP_KS = (2, 3, 20)
LAP_VECTORS = 4
ROUND = 10

COMBOS = {
    f"rrwp{k}+lap+ring+geom" + q: (f"rrwp{k}", "lap", "ring", "geom") + parts
    for k in RRWP_KS
    for q, parts in (("", ()), ("+afm+fm", ("afm", "fm")))
} | {"afm+fm": ("afm", "fm")}


def rrwp_matrices(A: np.ndarray, K: int) -> np.ndarray:
    """``[K, n, n]`` stack of ``M^k``, ``M = D^-1 A``, k = 1..K."""
    A_f = A.astype(np.float64)
    deg = A_f.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    M = deg_inv[:, None] * A_f
    out = np.empty((K, A.shape[0], A.shape[0]))
    cur = M.copy()
    for k in range(K):
        out[k] = cur
        cur = cur @ M
    return out


def lap_eigvecs(A: np.ndarray, n_vec: int) -> np.ndarray:
    """``[n, n_vec]`` eigenvectors ``phi_2..phi_{1+n_vec}`` of the normalised Laplacian,
    sign fixed by making the largest-magnitude entry positive."""
    n = A.shape[0]
    A_f = A.astype(np.float64)
    deg = A_f.sum(axis=1)
    dis = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(n) - (dis[:, None] * A_f) * dis[None, :]
    L = 0.5 * (L + L.T)
    _, evecs = np.linalg.eigh(L)
    out = np.zeros((n, n_vec))
    for k in range(min(n_vec, n - 1)):
        v = evecs[:, k + 1]  # skip the constant phi_1
        imax = int(np.argmax(np.abs(v)))
        out[:, k] = v if v[imax] >= 0 else -v
    return out


def ring_membership(src, dst, n: int, ring_sizes=RING_SIZES) -> np.ndarray:
    """``[n, len(ring_sizes)]`` 0/1: node lies on a cycle-basis cycle of that size."""
    feat = np.zeros((n, len(ring_sizes)), dtype=np.float32)
    if n == 0 or len(src) == 0:
        return feat
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(np.asarray(src).tolist(), np.asarray(dst).tolist()))
    col = {int(rs): i for i, rs in enumerate(ring_sizes)}
    for cycle in nx.cycle_basis(G):
        c = col.get(len(cycle))
        if c is not None:
            feat[cycle, c] = 1.0
    return feat


def pool1(X: np.ndarray) -> np.ndarray:
    """``[n, d]`` -> per-column quantiles and mean, ``[10 d]``."""
    X = np.sort(X, axis=0)
    return np.concatenate([np.quantile(X, QUANTILES, axis=0).ravel(), X.mean(0)])


def pool2(T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``[n, n, d]`` -> :func:`pool1` over off-diagonal pairs, then over edges, ``[20 d]``."""
    n = T.shape[0]
    return np.concatenate([pool1(T[~np.eye(n, dtype=bool)]), pool1(T[A > 0])])


def geometry(pos: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Register-geometry family from atom positions ``[n, 2]`` (um)."""
    n = len(pos)
    ii = np.arange(n)
    r = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    rs = r.copy()
    rs[ii, ii] = 1.0
    inv6 = rs**-6.0
    inv6[ii, ii] = 0.0
    tb = np.stack([r, np.log(rs), inv6], -1)
    tb[ii, ii] = 0.0
    ro = r + np.diag(np.full(n, np.inf))
    cen = np.linalg.norm(pos - pos.mean(0), axis=1)
    ob = np.stack([inv6.sum(1), inv6.max(1), ro.min(1), rs.sum(1) / (n - 1), cen], 1)
    return np.concatenate([pool1(ob), pool2(tb, A)])


def quench_features(occupation: np.ndarray, corr: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Pooled quench family from time-major ``occupation [T, n]`` and ``corr [T, n, n]``.

    Both are transposed explicitly to node-major (``[n, T]``, ``[n, n, T]``); the layout is
    never inferred from the shapes (a 10-rung ladder has n = T = 20).
    """
    return np.concatenate([pool1(occupation.T), pool2(corr.transpose(1, 2, 0), A)])


def graph_features(m: int, crossings, pos: np.ndarray, afm=None, fm=None) -> dict:
    """All base families of one ladder.

    ``pos`` = atom register ``[2m, 2]`` (um); ``afm`` / ``fm`` = optional
    ``(occupation [T, n], C [T, n, n])`` in the occupation convention
    (``C = <n_i n_j> - <n_i><n_j>``, not the ZZ ``4 C``).
    """
    G = ladder(m, crossings)
    n = 2 * m
    A = nx.to_numpy_array(G, nodelist=range(n))
    F = {"n": np.array([n], float)}
    P = rrwp_matrices(A.astype(int), max(RRWP_KS)).transpose(1, 2, 0)  # [n, n, K]
    diag = P[np.arange(n), np.arange(n)]  # [n, K]
    for k in RRWP_KS:
        F[f"rrwp{k}"] = np.concatenate([pool1(diag[:, :k]), pool2(P[:, :, :k], A)])

    deg = A.sum(1)
    L = np.diag(deg) - A
    Dm = 1 / np.sqrt(deg)
    Ln = np.eye(n) - Dm[:, None] * A * Dm[None, :]
    V = np.abs(lap_eigvecs(A.astype(int), LAP_VECTORS))
    F["lap"] = np.concatenate(
        [pool1(np.linalg.eigvalsh(L)[:, None]), pool1(np.linalg.eigvalsh(Ln)[:, None]), pool1(V)]
    )

    src, dst = np.nonzero(A)
    R = ring_membership(src, dst, n)
    F["ring"] = np.concatenate([pool1(R), R.sum(0)])
    F["geom"] = geometry(np.asarray(pos, dtype=np.float64), A)

    for name, q in (("afm", afm), ("fm", fm)):
        if q is not None:
            F[name] = quench_features(np.asarray(q[0]), np.asarray(q[1]), A)

    occ, C, _ = gs_correlations_dp(G)
    F["gs"] = np.concatenate([pool1(occ[:, None]), pool2(C[:, :, None], A)])
    F["gs_max"] = np.array([np.abs(C[~np.eye(n, dtype=bool)]).max()])
    return {k: np.round(v, ROUND) for k, v in F.items()}


def add_combos(F: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add the concatenated families of :data:`COMBOS` whose parts are present (in place)."""
    for name, parts in COMBOS.items():
        if all(p in F for p in parts):
            F[name] = np.hstack([F[p] for p in parts])
    return F
