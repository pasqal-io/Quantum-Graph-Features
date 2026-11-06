"""Turn raw bitstring counts into QGF datasets (onebody/twobody features, ZZ convention)."""

import numpy as np
import torch
from torch_geometric.data import Data

from qgf.generate.correlations import (
    QPU_READOUT_EPSILONS,
    ZZ_SCALE,
    correlation_from_arrays,
    jackknife_correlation_from_arrays,
    split_features,
)
from qgf.generate.raw import RawResults


def raw_correlations(
    raw: RawResults,
    g: int,
    jackknife: bool = False,
    epsilons: tuple[float, float] | None = QPU_READOUT_EPSILONS,
) -> np.ndarray:
    """``C [T, N, N]`` of graph index ``g``: plug-in estimator (emulation datasets) or
    readout-corrected jackknife (QPU dataset, see :func:`jackknife_correlation_from_counts`)."""
    out = []
    for t in range(len(raw.times_ns)):
        bits, w = raw.bit_matrix(g, t)
        if jackknife:
            out.append(jackknife_correlation_from_arrays(bits, w, epsilons)[0])
        else:
            out.append(correlation_from_arrays(bits, w))
    return np.stack(out)


def raw_features(raw: RawResults, g: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """``(onebody [T, N], twobody [T, N, N])`` in ZZ convention (``4 * C``)."""
    return split_features(raw_correlations(raw, g, **kwargs), ZZ_SCALE)


def attach_features(graphs: list[Data], raws: list[RawResults], **kwargs) -> list[Data]:
    """Copy ``graphs`` with features recomputed from ``raws`` (matched by ``initial_id``).

    Several ``raws`` (e.g. one QPU run per measurement time) are concatenated along time.
    """
    out = []
    for data in graphs:
        gid = int(data.initial_id.flatten()[0])
        parts = [raw_features(r, r.index_of(gid), **kwargs) for r in raws]
        new = data.clone()
        new.onebody_features = torch.from_numpy(np.concatenate([p[0] for p in parts])).float()
        new.twobody_features = torch.from_numpy(np.concatenate([p[1] for p in parts])).float()
        out.append(new)
    return out
