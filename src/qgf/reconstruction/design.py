"""Column blocks, pooling and design cache.

The methodological principle of the experiment is **pooling**: all pairs (i, j) of all
graphs are stacked in a single regression with one shared coefficient vector. A
:class:`DesignBlock` carries the design matrix and, per column, its *order* (the smallest
K for which the column exists). Designs are built once at ``k_max`` and sliced by order,
which makes the K grid almost free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from qgf.reconstruction.config import BasisContext
from qgf.reconstruction.data import GraphSample

LOGGER = logging.getLogger(__name__)

#: K-independent families: all their columns have order 0 and survive any truncation.
ORDER_ALWAYS: int = 0

#: Marker of columns valid at every time step.
TIME_STATIC: int = -1


@dataclass(frozen=True)
class DesignBlock:
    """Design matrix and column metadata.

    Attributes:
        matrix: (n_rows, n_cols) descriptor values.
        labels: name of each column.
        orders: (n_cols,) smallest K for which the column exists.
        time_index: (n_cols,) time step the column is reserved to, or ``TIME_STATIC``.
            Allows features aligned with the target time, such as the exact two-atom
            correlation.
    """

    matrix: np.ndarray
    labels: tuple[str, ...]
    orders: np.ndarray
    time_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"Design of rank {self.matrix.ndim}, expected 2")
        if len(self.labels) != self.matrix.shape[1]:
            raise ValueError(f"{len(self.labels)} labels for {self.matrix.shape[1]} columns")
        if self.orders.shape != (self.matrix.shape[1],):
            raise ValueError("orders must have one entry per column")
        if self.time_index is None:
            object.__setattr__(
                self,
                "time_index",
                np.full(self.matrix.shape[1], TIME_STATIC, dtype=np.int64),
            )
        elif self.time_index.shape != (self.matrix.shape[1],):
            raise ValueError("time_index must have one entry per column")

    @property
    def num_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_columns(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def has_time_columns(self) -> bool:
        """True if the block carries columns reserved to one time step."""
        return bool(np.any(self.time_index != TIME_STATIC))

    def _subset(self, keep: np.ndarray) -> DesignBlock:
        return DesignBlock(
            matrix=np.ascontiguousarray(self.matrix[:, keep]),
            labels=tuple(self.labels[c] for c in keep),
            orders=self.orders[keep],
            time_index=self.time_index[keep],
        )

    def select(self, k: int) -> DesignBlock:
        """Sub-block restricted to columns of order <= k."""
        return self._subset(np.flatnonzero(self.orders <= k))

    def select_time(self, step: int) -> DesignBlock:
        """Sub-block valid at time step ``step`` (static columns + columns of that step)."""
        return self._subset(
            np.flatnonzero((self.time_index == TIME_STATIC) | (self.time_index == step)),
        )

    def with_matrix(self, matrix: np.ndarray) -> DesignBlock:
        """Same column metadata, other rows."""
        return replace(self, matrix=matrix)


def concat_blocks(blocks: Sequence[DesignBlock]) -> DesignBlock:
    """Concatenates blocks column-wise (same rows, same order)."""
    if not blocks:
        raise ValueError("No block to concatenate")
    if len({block.num_rows for block in blocks}) != 1:
        raise ValueError("Blocks of different heights")
    return DesignBlock(
        matrix=np.concatenate([block.matrix for block in blocks], axis=1),
        labels=tuple(label for block in blocks for label in block.labels),
        orders=np.concatenate([block.orders for block in blocks]),
        time_index=np.concatenate([block.time_index for block in blocks]),
    )


def stack_blocks(blocks: Sequence[DesignBlock]) -> DesignBlock:
    """Stacks blocks row-wise (same columns): pooling over graphs."""
    if not blocks:
        raise ValueError("No block to stack")
    reference = blocks[0]
    for block in blocks[1:]:
        if block.labels != reference.labels:
            raise ValueError("Heterogeneous columns across graphs")
    return DesignBlock(
        matrix=np.concatenate([block.matrix for block in blocks], axis=0),
        labels=reference.labels,
        orders=reference.orders,
        time_index=reference.time_index,
    )


def with_constant(block: DesignBlock) -> DesignBlock:
    """Adds an explicit constant column.

    Only off-diagonal entries are regressed and (A^0)_ij = 0 for i != j, so the k=0
    column of the walk basis is identically zero: the design has no effective intercept.
    """
    ones = np.ones((block.num_rows, 1), dtype=float)
    constant = DesignBlock(matrix=ones, labels=("const",), orders=np.array([ORDER_ALWAYS]))
    return concat_blocks([constant, block])


def column_scales(matrix: np.ndarray) -> np.ndarray:
    """Per-column normalisation scales: max(max_p |X_pk|, 1)."""
    return np.maximum(np.abs(matrix).max(axis=0), 1.0)


def normalize_columns(matrix: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """Column normalisation applied **after** pooling.

    Required: (A^k)_ij grows as O(lambda_max^k) and overflows float64 around K ~ 30 on
    dense graphs.
    """
    if scales is None:
        scales = column_scales(matrix)
    return matrix / scales


def check_finite(block: DesignBlock, context: str) -> None:
    """Overflow guard for matrix powers."""
    if not np.all(np.isfinite(block.matrix)):
        bad = [block.labels[c] for c in np.flatnonzero(~np.isfinite(block.matrix).all(axis=0))]
        raise FloatingPointError(
            f"{context}: non-finite columns {bad[:5]}; reduce k_max",
        )


# --------------------------------------------------------------------------------------
# Family registry
# --------------------------------------------------------------------------------------
BasisBuilder = Callable[[GraphSample, BasisContext], DesignBlock]
_REGISTRY: dict[str, BasisBuilder] = {}
#: Group of each family, for traceability of the reports.
_PHASES: dict[str, str] = {}
#: Families whose design depends on the quench and the time grid: their cache must be
#: keyed accordingly, otherwise the AFM design would be read back for the FM run.
_QUENCH_SENSITIVE: set[str] = set()


def register_basis(
    name: str,
    phase: str,
    quench_sensitive: bool = False,
) -> Callable[[BasisBuilder], BasisBuilder]:
    """Registration decorator of a descriptor family."""

    def wrap(builder: BasisBuilder) -> BasisBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Family already registered: {name}")
        _REGISTRY[name] = builder
        _PHASES[name] = phase
        if quench_sensitive:
            _QUENCH_SENSITIVE.add(name)
        return builder

    return wrap


def available_bases() -> dict[str, str]:
    """Family name -> group."""
    return dict(_PHASES)


def is_quench_sensitive(atom: str) -> bool:
    """Does the design of this family depend on the quench and the times?"""
    return atom in _QUENCH_SENSITIVE


def resolve_family(family: str) -> tuple[str, ...]:
    """Splits "walk+laplacian" into validated atoms."""
    atoms = tuple(part.strip() for part in family.split("+") if part.strip())
    if not atoms:
        raise ValueError(f"Empty family: {family!r}")
    unknown = [atom for atom in atoms if atom not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown families {unknown}; available: {sorted(_REGISTRY)}")
    return atoms


def build_sample_block(sample: GraphSample, atom: str, context: BasisContext) -> DesignBlock:
    """Builds the block of one family atom for one graph."""
    block = _REGISTRY[atom](sample, context)
    if block.num_rows != sample.num_pairs:
        raise ValueError(
            f"Family {atom}: {block.num_rows} rows for {sample.num_pairs} pairs "
            f"of graph {sample.graph_id}",
        )
    return block


def build_sample_design(
    sample: GraphSample,
    family: str,
    context: BasisContext,
) -> DesignBlock:
    """Design of one graph for a composite family."""
    atoms = resolve_family(family)
    return concat_blocks([build_sample_block(sample, atom, context) for atom in atoms])


# --------------------------------------------------------------------------------------
# Pooling and disk cache
# --------------------------------------------------------------------------------------


def pooled_groups(samples: Sequence[GraphSample]) -> np.ndarray:
    """(P,) graph identifier of every pooled row."""
    return np.concatenate(
        [np.full(sample.num_pairs, sample.graph_id, dtype=np.int64) for sample in samples],
    )


def pooled_targets(samples: Sequence[GraphSample]) -> np.ndarray:
    """Pooled target (P, N_t)."""
    return np.concatenate([sample.targets() for sample in samples], axis=0)


def build_pooled_block(
    samples: Iterable[GraphSample],
    atom: str,
    context: BasisContext,
) -> DesignBlock:
    """Pooled design (P, n_cols) of one family atom, rows in graph order."""
    return stack_blocks([build_sample_block(sample, atom, context) for sample in samples])


def _context_fingerprint(context: BasisContext, atom: str) -> str:
    """Hash of the context fields that actually influence the design."""
    payload: dict[str, object] = {"atom": atom, "k_max": context.k_max}
    fields = {
        "r_nn_um": context.r_nn_um,
        "c6": context.c6,
        "rbf_centers": context.rbf_centers,
        "rbf_range_rnn": context.rbf_range_rnn,
        "beta_count": context.beta_count,
        "beta_mode": context.beta_mode,
        "beta_max_fraction": context.beta_max_fraction,
        "max_ring_size": context.max_ring_size,
        "heat_taus": context.heat_taus,
        "max_geodesic": context.max_geodesic,
    }
    fields["sign_tie_decimals"] = context.sign_tie_decimals
    payload.update(fields)
    if is_quench_sensitive(atom):
        payload["quench"] = (
            context.quench.name,
            context.quench.omega_over_jnn,
            context.quench.delta_mode,
        )
        payload["times_ns"] = context.times_ns
        payload["twa"] = (context.twa_trajectories, context.twa_dt_us, context.twa_seed)
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def cached_pooled_block(
    samples: Sequence[GraphSample],
    atom: str,
    context: BasisContext,
    dataset: str,
    cache_dir: str | None = None,
    max_mb: float = 512.0,
) -> DesignBlock:
    """Pooled design with a disk cache keyed by (dataset, family, k_max, context).

    The time alignment (``time_index``) is stored with the matrix. The original code did
    not store it, so a design read back from the cache silently became static.
    """
    if cache_dir is None:
        return build_pooled_block(samples, atom, context)

    fingerprint = _context_fingerprint(context, atom)
    path = os.path.join(
        cache_dir,
        dataset,
        f"{atom}_k{context.k_max}_n{len(samples)}_{fingerprint}.npz",
    )
    if os.path.exists(path):
        LOGGER.info("design %s read from cache (%s)", atom, path)
        with np.load(path, allow_pickle=False) as stored:
            return DesignBlock(
                matrix=stored["matrix"],
                labels=tuple(str(label) for label in stored["labels"]),
                orders=stored["orders"],
                time_index=stored["time_index"],
            )

    block = build_pooled_block(samples, atom, context)
    size_mb = block.matrix.nbytes / 1e6
    if size_mb > max_mb:
        LOGGER.warning("design %s not cached: %.0f MB > %.0f MB", atom, size_mb, max_mb)
        return block
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        matrix=block.matrix,
        labels=np.asarray(block.labels),
        orders=block.orders,
        time_index=block.time_index,
    )
    return block


def build_pooled_design(
    samples: Sequence[GraphSample],
    family: str,
    context: BasisContext,
    dataset: str,
    cache_dir: str | None = None,
    max_mb: float = 512.0,
) -> DesignBlock:
    """Pooled design of a composite family, atom by atom (each cached separately)."""
    atoms = resolve_family(family)
    blocks = [
        cached_pooled_block(samples, atom, context, dataset, cache_dir, max_mb) for atom in atoms
    ]
    design = concat_blocks(blocks)
    check_finite(design, f"family {family}")
    return design
