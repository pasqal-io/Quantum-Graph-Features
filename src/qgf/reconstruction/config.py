"""Configuration of the QGF reconstruction-ceiling experiment.

A run is fully described by a :class:`RunConfig`: dataset, quenches, descriptor families,
K and t grids, probes, seed. Physical constants and grids are gathered here so that any
deviation from the paper is visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

# --------------------------------------------------------------------------------------
# Time grids
# --------------------------------------------------------------------------------------
# The released datasets store their true time grid in ``meta["times_ns"]``
# (``linspace(100, 4000, 20, dtype=int)`` ns for MUTAG / ZINC emulations).
DATA_TIMES_NS: np.ndarray = np.linspace(100, 4000, 20, dtype=int).astype(float)

#: Grid index of the reference instant of the tables (Tables 14/15, extended ceiling):
#: stored time index 7, i.e. 1536 ns on the data grid.
REFERENCE_TIME_INDEX: int = 7

# --------------------------------------------------------------------------------------
# Physical constants of the device
# --------------------------------------------------------------------------------------
#: Nearest-neighbour spacing of the embeddings, in um.
R_NN_UM: float = 5.01
#: C6/hbar in um^6 rad/us.
C6_RAD_UM6_PER_US: float = 8.66e5

#: Conversion of released correlations (ZZ convention, ``4 C``) to the occupation-number
#: convention ``C = <n_i n_j> - <n_i><n_j>`` in [-1/4, 1/4]. R^2 is invariant under this
#: global factor; the physical scale is not.
ZZ_TO_OCCUPATION_SCALE: float = 0.25


@dataclass(frozen=True)
class QuenchSpec:
    """Quench parameters, in units of J_NN (AFM) or of delta_infinity (FM)."""

    name: str
    omega_over_jnn: float
    delta_mode: Literal["zero", "two_delta_inf"]

    def omega(self, j_nn: float) -> float:
        """Omega in rad/us for a nearest-neighbour coupling ``j_nn`` in rad/us."""
        return self.omega_over_jnn * j_nn

    def delta(self, interaction: np.ndarray) -> float:
        """Delta in rad/us from the interaction matrix (rad/us), zero diagonal.

        delta_inf = sum_{i!=j} J_ij / (4 N); the FM regime uses delta = 2 delta_inf.
        """
        if self.delta_mode == "zero":
            return 0.0
        num_nodes = interaction.shape[0]
        delta_inf = interaction.sum() / (4.0 * num_nodes)
        return 2.0 * delta_inf


AFM_QUENCH = QuenchSpec(name="AFM", omega_over_jnn=0.3, delta_mode="zero")
FM_QUENCH = QuenchSpec(name="FM", omega_over_jnn=0.5, delta_mode="two_delta_inf")
QUENCHES: dict[str, QuenchSpec] = {"AFM": AFM_QUENCH, "FM": FM_QUENCH}

# --------------------------------------------------------------------------------------
# Sweep grids
# --------------------------------------------------------------------------------------
K_GRID_OLS: tuple[int, ...] = (1, 3, 5, 7, 10, 14, 21, 30, 50)
K_GRID_NONLINEAR: tuple[int, ...] = (3, 7, 21)
#: Reference K of the extension protocol.
K_EXTENSION: int = 21


@dataclass(frozen=True)
class BasisContext:
    """Hyperparameters shared by the basis builders.

    ``k_max`` is the largest order built: designs are built once at ``k_max`` and then
    sliced by column order for every K of the grid.
    """

    k_max: int = max(K_GRID_OLS)
    quench: QuenchSpec = AFM_QUENCH
    r_nn_um: float = R_NN_UM
    c6: float = C6_RAD_UM6_PER_US
    # Geometry basis
    rbf_centers: int = 16
    rbf_range_rnn: tuple[float, float] = (0.5, 4.0)
    # Mean-field Ising basis
    beta_count: int = 8
    beta_mode: Literal["relative", "absolute"] = "relative"
    beta_max_fraction: float = 0.9
    # Ring basis
    max_ring_size: int = 8
    # Classical spin dynamics basis (dTWA on the same Hamiltonian).
    #: Number of trajectories sampled from the discrete Wigner distribution of |0...0>.
    twa_trajectories: int = 512
    #: RK4 step in us; must satisfy Omega dt << 1.
    twa_dt_us: float = 0.01
    #: Seed of the stochastic initial conditions.
    twa_seed: int = 0
    #: Rounding of eigenvector magnitudes before fixing their sign (makes the
    #: tie-breaking independent of the numpy/LAPACK build).
    sign_tie_decimals: int = 10
    # Global descriptors
    heat_taus: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    max_geodesic: int = 10
    #: Times (ns) of the time-dependent two-body features.
    times_ns: tuple[float, ...] = tuple(DATA_TIMES_NS)


@dataclass(frozen=True)
class ProbeSpec:
    """A probe and the (K, t) sub-domain it is evaluated on.

    ``time_indices=None`` means the whole time grid.
    """

    name: Literal["ols", "kernel_ridge", "random_forest"]
    k_grid: tuple[int, ...] = K_GRID_OLS
    time_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RunConfig:
    """Complete description of a run (one config file is dumped per run)."""

    dataset: str = "MUTAG"
    quenches: tuple[str, ...] = ("AFM", "FM")
    #: Families of the registry; "a+b" concatenates column blocks.
    families: tuple[str, ...] = ("walk", "rrwp")
    probes: tuple[ProbeSpec, ...] = (ProbeSpec(name="ols"),)
    #: Subset of time indices of the run; None = whole grid.
    time_indices: tuple[int, ...] | None = None
    #: OLS: add an explicit constant column.
    intercept: bool = False
    n_splits: int = 5
    seed: int = 0
    #: Brings released correlations (ZZ) to the occupation-number convention.
    target_scale: float = ZZ_TO_OCCUPATION_SCALE
    max_graphs: int | None = None
    #: None = automatic choice from the pooled size (streaming for ZINC).
    streaming: bool | None = None
    output_dir: str = "results/reconstruction/run"
    cache_dir: str = "results/cache/reconstruction"
    cache_designs: bool = True
    #: Max size (MB) of a design written to the disk cache.
    cache_max_mb: float = 512.0
    #: Root of the released datasets (None = ``qgf.data.DEFAULT_ROOT``).
    data_root: str | None = None
    context: BasisContext = field(default_factory=BasisContext)

    def times(self) -> np.ndarray:
        """Time grid of the run, in ns."""
        return DATA_TIMES_NS

    def context_for(self, quench: str) -> BasisContext:
        """Basis context specialised to a quench and to the time grid of the run."""
        if quench not in QUENCHES:
            raise ValueError(f"Unknown quench: {quench}")
        return replace(
            self.context,
            quench=QUENCHES[quench],
            times_ns=tuple(float(t) for t in self.times()),
        )


def nearest_time_index(times_ns: np.ndarray, target_ns: float) -> int:
    """Index of the grid point closest to ``target_ns``."""
    return int(np.argmin(np.abs(np.asarray(times_ns, dtype=float) - float(target_ns))))
