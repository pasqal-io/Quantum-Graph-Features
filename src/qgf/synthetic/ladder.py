"""Ladder graphs with crossings: builder, exact ground state, unit-disk atom register.

A ladder with ``m`` rungs has nodes ``T_i = i`` (top rail) and ``B_i = m + i`` (bottom
rail), rungs ``(T_i, B_i)`` and rails ``(T_i, T_{i+1})``, ``(B_i, B_{i+1})``. A
**crossing** at cell ``c`` adds the diagonal ``(T_c, B_{c+1})``.

A plain ladder has two maximum independent sets (the two alternations). A crossing is
compatible with only one of them: crossings whose cell parities agree **pin** a unique
ground state of the blockade (MIS) Hamiltonian, conflicting parities leave it
**frustrated** (degenerate). The label of the synthetic task is ``y = 1`` (pinned, unique
ground state) vs ``y = 0`` (frustrated).

The ground-state manifold is handled exactly by a transfer-matrix DP over rungs
(:func:`gs_correlations_dp`, cost O(m^2) whatever the degeneracy); :func:`gs_correlations`
is the brute-force reference for small graphs.
"""

import networkx as nx
import numpy as np

T_STATE, B_STATE, NONE = 0, 1, 2  # per-rung occupation choice
A_UM = 5.01  # nearest-neighbour atom distance (um), the paper's r_NN


def ladder(m: int, crossings=()) -> nx.Graph:
    """Ladder with ``m`` rungs; each crossing cell ``c`` adds the edge ``(T_c, B_{c+1})``."""
    G = nx.Graph()
    for i in range(m):
        G.add_edge(i, m + i)
        if i + 1 < m:
            G.add_edge(i, i + 1)
            G.add_edge(m + i, m + i + 1)
    for c in crossings:
        if not 0 <= c < m - 1:
            raise ValueError(f"crossing cell {c} out of range for m={m}")
        G.add_edge(c, m + c + 1)
    return G


def rung_data(G: nx.Graph) -> tuple[int, set[int]]:
    """``(m, crossing cells)`` of a graph built by :func:`ladder`."""
    m = G.number_of_nodes() // 2
    return m, {c for c in range(m - 1) if G.has_edge(c, m + c + 1)}


def _allowed(prev: int, cur: int, cell: int, cross: set[int]) -> bool:
    """Can rung ``cell + 1`` be in state ``cur`` given rung ``cell`` in state ``prev``?"""
    if prev == cur and prev != NONE:  # rail conflict
        return False
    # the crossing diagonal (T_cell, B_cell+1)
    return not (cell in cross and prev == T_STATE and cur == B_STATE)


def gs_correlations_dp(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact ``(occ [n], C [n, n], degeneracy)`` of the uniform distribution over the MIS.

    The uniform distribution over optimal rung-state sequences is a Markov chain along the
    ladder: ``Q_c[s, t] = E_c[s, t] N_{c+1}(t) / N_c(s)``, where ``E_c`` keeps only
    transitions on an optimal path and ``N_c(s)`` counts optimal suffixes. Pair marginals
    are ``p_a(s) [Q_a ... Q_{b-1}]_{s,t}``, so ``C`` costs O(m^2) 3x3 products.
    """
    m, cross = rung_data(G)
    gain = np.array([1, 1, 0])
    allowed = np.ones((m - 1, 3, 3), dtype=bool)
    for c in range(m - 1):
        for s in range(3):
            for t in range(3):
                allowed[c, s, t] = _allowed(s, t, c, cross)

    # best sizes: fwd[c, s] over rungs 0..c ending in s, bwd[c, s] over rungs c..m-1
    neg = -(10**9)
    fwd = np.full((m, 3), neg)
    bwd = np.full((m, 3), neg)
    fwd[0] = gain
    for c in range(m - 1):
        fwd[c + 1] = np.where(allowed[c], fwd[c][:, None], neg).max(0) + gain
    bwd[m - 1] = gain
    for c in range(m - 2, -1, -1):
        bwd[c] = np.where(allowed[c], bwd[c + 1][None, :], neg).max(1) + gain
    size = int(fwd[m - 1].max())

    # optimal transitions and suffix counts (python ints: degeneracy can be huge)
    opt = allowed & (fwd[:-1][:, :, None] + bwd[1:][:, None, :] == size)
    count_suffix = [[0] * 3 for _ in range(m)]
    for s in range(3):
        count_suffix[m - 1][s] = int(fwd[m - 1, s] == size)
    for c in range(m - 2, -1, -1):
        for s in range(3):
            count_suffix[c][s] = sum(count_suffix[c + 1][t] for t in range(3) if opt[c, s, t])
    start = [count_suffix[0][s] if bwd[0, s] == size else 0 for s in range(3)]
    count = sum(start)

    Q = np.zeros((m - 1, 3, 3))
    for c in range(m - 1):
        for s in range(3):
            if count_suffix[c][s]:
                for t in range(3):
                    if opt[c, s, t]:
                        Q[c, s, t] = count_suffix[c + 1][t] / count_suffix[c][s]
    p = np.zeros((m, 3))
    p[0] = [x / count for x in start]
    for c in range(m - 1):
        p[c + 1] = p[c] @ Q[c]

    n = 2 * m
    occ = np.concatenate([p[:, T_STATE], p[:, B_STATE]])
    pair = np.zeros((n, n))
    idx = np.arange(m)
    pair[idx, idx] = p[:, T_STATE]
    pair[m + idx, m + idx] = p[:, B_STATE]
    # joint J[a] = diag(p_a) Q_a ... Q_{b-1}, advanced for all a at once per gap d
    J = np.einsum("as,st->ast", p, np.eye(3))
    for d in range(1, m):
        a = np.arange(m - d)
        J = np.einsum("ast,atu->asu", J[: m - d], Q[a + d - 1])
        b = a + d
        for (s, oa), (t, ob) in [
            ((T_STATE, 0), (T_STATE, 0)),
            ((T_STATE, 0), (B_STATE, m)),
            ((B_STATE, m), (T_STATE, 0)),
            ((B_STATE, m), (B_STATE, m)),
        ]:
            pair[oa + a, ob + b] = J[:, s, t]
            pair[ob + b, oa + a] = J[:, s, t]
    return occ, pair - np.outer(occ, occ), count


def mis_bruteforce(G: nx.Graph) -> tuple[int, list[set[int]]]:
    """All maximum independent sets by bitmask enumeration (n <= 24)."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n > 24:
        raise ValueError("brute force only for n <= 24")
    edges = [(nodes.index(u), nodes.index(v)) for u, v in G.edges()]
    best, found = -1, []
    for mask in range(1 << n):
        if any((mask >> u) & 1 and (mask >> v) & 1 for u, v in edges):
            continue
        k = mask.bit_count()
        if k > best:
            best, found = k, [mask]
        elif k == best:
            found.append(mask)
    return best, [{nodes[i] for i in range(n) if (mk >> i) & 1} for mk in found]


def gs_correlations(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force ``(occ, C)`` over the uniform distribution on the MIS (small graphs)."""
    _, sets = mis_bruteforce(G)
    n = G.number_of_nodes()
    occ = np.zeros(n)
    pair = np.zeros((n, n))
    for S in sets:
        v = np.zeros(n)
        v[sorted(S)] = 1.0
        occ += v
        pair += np.outer(v, v)
    occ /= len(sets)
    pair /= len(sets)
    return occ, pair - np.outer(occ, occ)


def unit_disk_positions(G: nx.Graph, a: float = A_UM) -> np.ndarray:
    """Unit-disk atom register realising the ladder, spacing ``a`` (um): ``[2m, 2]``.

    The emulators derive couplings from ``C6 / r_ij^6``, so edges must be geometrically
    short and non-edges long. A plain cell is a square of side ``a`` (diagonals at
    ``a sqrt 2``, coupling ``J_NN / 8``). A crossing cell is a rhombus of two equilateral
    triangles: the crossing diagonal ``(T_c, B_{c+1})`` sits at ``a``, the opposite one at
    ``a sqrt 3`` (``J_NN / 27``). Rungs stay vertical, so a crossing advances the strip by
    ``(a sqrt 3 / 2, a / 2)`` instead of ``(a, 0)``. Rows ordered ``T_0..T_{m-1}, B_0..B_{m-1}``.
    """
    m, cross = rung_data(G)
    xs = np.zeros(m)
    ys = np.zeros(m)  # y of the bottom node of each rung
    for c in range(m - 1):
        if c in cross:
            xs[c + 1] = xs[c] + a * np.sqrt(3) / 2
            ys[c + 1] = ys[c] + a / 2
        else:
            xs[c + 1] = xs[c] + a
            ys[c + 1] = ys[c]
    pos = np.zeros((2 * m, 2))
    for i in range(m):
        pos[i] = (xs[i], ys[i] + a)
        pos[m + i] = (xs[i], ys[i])
    return pos


def verify_unit_disk(G: nx.Graph, a: float = A_UM, tol: float = 1.15):
    """``(ok, n_missing, n_extra, min_nonedge / a)``: is the register's unit-disk graph
    (pairs within ``tol * a``) exactly ``G``?"""
    pos = unit_disk_positions(G, a)
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    off = ~np.eye(n, dtype=bool)
    geo = (d <= tol * a) & off
    want = nx.to_numpy_array(G, nodelist=range(n)).astype(bool)
    missing = int((want & ~geo).sum() // 2)
    extra = int((geo & ~want).sum() // 2)
    nonedge = d[~want & off]
    return missing == 0 and extra == 0, missing, extra, float(nonedge.min() / a)


def ladder_layout(G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Drawing positions: rung ``i`` at ``x = i``, top rail at ``y = 1``."""
    m = G.number_of_nodes() // 2
    return {**{i: (i, 1.0) for i in range(m)}, **{m + i: (i, 0.0) for i in range(m)}}


def parse_crossings(text: str) -> tuple[int, ...]:
    """``"2,5"`` -> ``(2, 5)``; ``""`` -> ``()``."""
    return tuple(int(c) for c in str(text).split(",") if c)
