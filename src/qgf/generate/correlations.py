"""Connected correlations from bitstring counts.

``C_ij = <n_i n_j> - <n_i><n_j>`` with ``n_i`` the Rydberg occupation of atom ``i``.
Released features use the ZZ convention ``4 * C`` (``ZZ_SCALE``).

Counts are ``dict[str, int]`` mapping a bitstring (character ``i`` = atom ``i``,
``"1"`` = Rydberg) to the number of shots, as returned by pulser/emulators/QPU.
"""

from functools import reduce

import numpy as np

ZZ_SCALE = 4.0
# Readout errors of the QPU run: eps = P(obs 1 | true 0), eps' = P(obs 0 | true 1).
QPU_READOUT_EPSILONS = (0.025, 0.1)


def counts_to_arrays(counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bits [K, N] float, weights [K] float)`` for the unique bitstrings."""
    keys = list(counts)
    bits = np.array([[c == "1" for c in s] for s in keys], dtype=float)
    weights = np.array([counts[k] for k in keys], dtype=float)
    return bits, weights


def correlation_from_counts(counts: dict[str, int]) -> np.ndarray:
    """Plain (plug-in) connected correlation matrix ``C`` [N, N]."""
    return correlation_from_arrays(*counts_to_arrays(counts))


def correlation_from_arrays(bits: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Same as :func:`correlation_from_counts` from ``(bits [K, N], weights [K])``."""
    total = w.sum()
    avg_n = w @ bits / total
    avg_nn = bits.T @ (w[:, None] * bits) / total
    return avg_nn - np.outer(avg_n, avg_n)


def confusion_matrix_nqubit(n: int, eps: float, epsp: float) -> np.ndarray:
    """``A_nq = A_1q^{otimes n}`` such that ``p_obs = A_nq @ p_true``."""
    a1 = np.array([[1 - eps, epsp], [eps, 1 - epsp]], dtype=float)
    return reduce(np.kron, [a1] * n)


def _readout_coefficients(eps: float, epsp: float) -> tuple[float, ...]:
    """Coefficients of the readout-corrected connected correlation.

    ``C_true = c0 + ci <n_i> + cj <n_j> + cij <n_i n_j> + cicj <n_i><n_j>``
    (observed moments on the right), from the inverse 1- and 2-qubit confusion matrices.
    """
    a1 = np.linalg.inv(confusion_matrix_nqubit(1, eps, epsp))
    a2 = np.linalg.inv(confusion_matrix_nqubit(2, eps, epsp))
    c0 = a2[3, 0] - a1[1, 0] ** 2
    ci = (a2[3, 2] - a2[3, 0]) - a1[1, 0] * (a1[1, 1] - a1[1, 0])
    cj = (a2[3, 1] - a2[3, 0]) - a1[1, 0] * (a1[1, 1] - a1[1, 0])
    cij = a2[3, 0] - a2[3, 1] - a2[3, 2] + a2[3, 3]
    cicj = -((a1[1, 1] - a1[1, 0]) ** 2)
    return c0, ci, cj, cij, cicj


def jackknife_correlation_from_counts(
    counts: dict[str, int],
    epsilons: tuple[float, float] | None = QPU_READOUT_EPSILONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bias-corrected weighted-jackknife estimate of ``C`` and its standard error.

    Leave-one-shot-out over the unique bitstrings (weighted by their counts), with
    optional readout-error correction ``epsilons = (eps, eps')``; ``None`` disables it.
    Off-diagonal entries use the two-atom readout correction; the diagonal ``C_ii`` uses
    the single-atom correction (``n_i^2 = n_i``).
    Returns ``(C [N, N], std_error [N, N])``.
    """
    return jackknife_correlation_from_arrays(*counts_to_arrays(counts), epsilons=epsilons)


def jackknife_correlation_from_arrays(
    bits: np.ndarray,
    w: np.ndarray,
    epsilons: tuple[float, float] | None = QPU_READOUT_EPSILONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as :func:`jackknife_correlation_from_counts` from ``(bits [K, N], weights [K])``."""
    total = w.sum()
    if total <= 1:
        raise ValueError("at least two shots are needed for the jackknife")
    if epsilons is None:
        c0, ci, cj, cij, cicj = 0.0, 0.0, 0.0, 1.0, -1.0
    else:
        eps, epsp = (float(e) for e in epsilons)
        if not (0.0 <= eps <= 1.0 and 0.0 <= epsp <= 1.0):
            raise ValueError("eps and epsp must be in [0, 1]")
        c0, ci, cj, cij, cicj = _readout_coefficients(eps, epsp)

    def g(nn, ni, nj):
        return c0 + ci * ni + cj * nj + cij * nn + cicj * ni * nj

    sum_n = w @ bits  # [N]
    sum_nn = bits.T @ (w[:, None] * bits)  # [N, N]
    outer = bits[:, :, None] * bits[:, None, :]  # [K, N, N]
    red_n = (sum_n[None, :] - bits) / (total - 1)  # [K, N]
    red_nn = (sum_nn[None] - outer) / (total - 1)  # [K, N, N]
    reduced = g(red_nn, red_n[:, :, None], red_n[:, None, :])  # [K, N, N]
    mean_n = sum_n / total
    full = g(sum_nn / total, mean_n[:, None], mean_n[None, :])

    if epsilons is not None:
        # The pair formula assumes independent readout of i and j, which is wrong for
        # i == j (n_i^2 = n_i): correct the occupation and use C_ii = p (1 - p) instead.
        a1 = np.linalg.inv(confusion_matrix_nqubit(1, eps, epsp))

        def g_diag(ni):
            p = a1[1, 0] + (a1[1, 1] - a1[1, 0]) * ni
            return p - p**2

        idx = np.arange(bits.shape[1])
        reduced[:, idx, idx] = g_diag(red_n)
        full[idx, idx] = g_diag(mean_n)

    simple = np.tensordot(w, reduced, axes=1) / total
    var = np.tensordot(w, (reduced - simple) ** 2, axes=1) * (total - 1) / total
    estimate = total * full - (total - 1) * simple
    return estimate, np.sqrt(var)


def split_features(corr: np.ndarray, zz_scale: float = ZZ_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """``corr [T, N, N]`` -> ``(onebody [T, N], twobody [T, N, N] with zero diagonal)``, scaled."""
    corr = zz_scale * np.asarray(corr, dtype=float)
    onebody = np.diagonal(corr, axis1=1, axis2=2).copy()
    twobody = corr.copy()
    idx = np.arange(corr.shape[1])
    twobody[:, idx, idx] = 0.0
    return onebody, twobody
