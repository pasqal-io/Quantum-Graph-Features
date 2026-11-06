"""Base descriptor families: walk powers, RRWP, Laplacian augmentation.

Two families indexed by k = 0..K:

    phi^walk_k(A)_ij = (A^k)_ij       phi^rrwp_k(A)_ij = ((D^-1 A)^k)_ij

plus the Laplacian augmentation: the first K eigenpairs {u_k(i), u_k(j), lambda_k} of the
symmetric normalised Laplacian L = I - D^-1/2 A D^-1/2, i.e. 3K columns.
"""

from __future__ import annotations

import numpy as np

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample
from qgf.reconstruction.design import DesignBlock, register_basis


def walk_powers(
    matrix: np.ndarray,
    k_max: int,
    pairs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Powers 0..k_max of ``matrix`` restricted to the pairs (i<j).

    Returns:
        (n_pairs, k_max + 1); column k holds (matrix^k)_ij.
    """
    num_nodes = matrix.shape[0]
    columns = []
    power = np.eye(num_nodes, dtype=float)
    for _ in range(k_max + 1):
        columns.append(power[pairs])
        power = power @ matrix
    return np.stack(columns, axis=1)


def _orders(k_max: int) -> np.ndarray:
    return np.arange(k_max + 1, dtype=np.int64)


@register_basis("walk", phase="P1")
def walk_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Adjacency powers (A^k)_ij for k = 0..k_max."""
    matrix = walk_powers(
        np.asarray(sample.adjacency, dtype=float),
        context.k_max,
        sample.pair_index,
    )
    return DesignBlock(
        matrix=matrix,
        labels=tuple(f"walk_{k}" for k in range(context.k_max + 1)),
        orders=_orders(context.k_max),
    )


def random_walk_matrix(adjacency: np.ndarray) -> np.ndarray:
    """D^-1 A, guarded against zero-degree nodes."""
    adjacency = np.asarray(adjacency, dtype=float)
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(degree, 1.0)


@register_basis("rrwp", phase="P1")
def rrwp_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Random-walk powers ((D^-1 A)^k)_ij for k = 0..k_max."""
    matrix = walk_powers(
        random_walk_matrix(sample.adjacency),
        context.k_max,
        sample.pair_index,
    )
    return DesignBlock(
        matrix=matrix,
        labels=tuple(f"rrwp_{k}" for k in range(context.k_max + 1)),
        orders=_orders(context.k_max),
    )


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """L = I - D^-1/2 A D^-1/2, chosen for its bounded spectrum [0, 2].

    The bounded spectrum makes eigenvalues comparable across graph sizes, which the
    combinatorial Laplacian does not.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    degree = adjacency.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    return np.eye(adjacency.shape[0]) - (adjacency * inv_sqrt[:, None]) * inv_sqrt[None, :]


#: Rounding of eigenvector magnitudes before choosing the sign-fixing entry.
SIGN_TIE_DECIMALS: int = 10


def canonical_eigenvectors(
    laplacian: np.ndarray,
    tie_decimals: int = SIGN_TIE_DECIMALS,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs sorted by increasing lambda, with canonicalised signs.

    The sign of each eigenvector is fixed by its largest-magnitude entry: an ad hoc
    treatment of the sign ambiguity. Magnitudes are rounded before the argmax so that
    exact ties (e.g. entries +-1/sqrt(2) of symmetric molecules) resolve to the lowest
    index instead of to LAPACK rounding noise, independently of the numpy/LAPACK build.
    """
    values, vectors = np.linalg.eigh(laplacian)
    num_nodes = vectors.shape[1]
    magnitude = np.round(np.abs(vectors), tie_decimals)
    dominant = np.argmax(magnitude, axis=0)
    signs = np.sign(vectors[dominant, np.arange(num_nodes)])
    signs[signs == 0] = 1.0
    return values, vectors * signs


def laplacian_features(
    adjacency: np.ndarray,
    k_max: int,
    pairs: tuple[np.ndarray, np.ndarray],
    tie_decimals: int = SIGN_TIE_DECIMALS,
) -> np.ndarray:
    """(n_pairs, 3 k_max): [u_k(i)], [u_k(j)], [lambda_k], zero-padded if n < k_max.

    The first k_max eigenpairs in increasing lambda are kept, **trivial pair included**
    (lambda_0 = 0). Labels are therefore numbered from 1: ``lap_ui_1`` is pair index 0.
    """
    num_nodes = adjacency.shape[0]
    values, vectors = canonical_eigenvectors(normalized_laplacian(adjacency), tie_decimals)
    kept = min(k_max, num_nodes)
    padded_values = np.zeros(k_max)
    padded_values[:kept] = values[:kept]
    padded_vectors = np.zeros((num_nodes, k_max))
    padded_vectors[:, :kept] = vectors[:, :kept]
    rows, cols = pairs
    return np.concatenate(
        [
            padded_vectors[rows],
            padded_vectors[cols],
            np.tile(padded_values, (rows.size, 1)),
        ],
        axis=1,
    )


@register_basis("laplacian", phase="P1")
def laplacian_basis(sample: GraphSample, context: BasisContext) -> DesignBlock:
    """Laplacian augmentation: 3 k_max columns (u_k(i), u_k(j), lambda_k)."""
    k_max = context.k_max
    matrix = laplacian_features(
        sample.adjacency,
        k_max,
        sample.pair_index,
        context.sign_tie_decimals,
    )
    labels = (
        tuple(f"lap_ui_{k}" for k in range(1, k_max + 1))
        + tuple(f"lap_uj_{k}" for k in range(1, k_max + 1))
        + tuple(f"lap_lambda_{k}" for k in range(1, k_max + 1))
    )
    orders = np.tile(np.arange(1, k_max + 1, dtype=np.int64), 3)
    return DesignBlock(matrix=matrix, labels=labels, orders=orders)
