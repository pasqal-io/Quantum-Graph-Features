"""Emulation of a sequence with emu-sv (N <= 24) or emu-mps (N > 24), sampling bitstrings."""

import logging

import pulser
from emu_mps import MPSBackend, MPSConfig
from emu_sv import SVBackend, SVConfig
from pulser.backend import BitStrings, CorrelationMatrix, Occupation

SV_MAX_QUBITS = 24
DT_NS = 10
MPS_MAX_BOND_DIM = 1024


def make_config(
    n_qubits: int,
    observables: list,
    use_gpu: bool = True,
    verbose: bool = False,
    with_modulation: bool = False,
    *,
    dt: float = DT_NS,
    sv_max_qubits: int = SV_MAX_QUBITS,
    max_bond_dim: int = MPS_MAX_BOND_DIM,
    interaction_matrix=None,
):
    """Configuration of the paper's runs: dt = 10 ns, no modulation, MPS bond dim <= 1024.

    The keyword-only overrides serve the synthetic ladder runs (emu-mps for every size,
    bond dimension 16, dt = 20 ns) and the non-unit-disk runs (``interaction_matrix``
    [N, N] replaces the register's ``C6 / r^6``); the defaults are the MUTAG / ZINC settings.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    extra = {}
    if interaction_matrix is not None:
        extra["interaction_matrix"] = [[float(v) for v in row] for row in interaction_matrix]
    if n_qubits <= sv_max_qubits:
        return SVBackend, SVConfig(
            dt=dt,
            observables=observables,
            log_level=log_level,
            with_modulation=with_modulation,
            gpu=None if use_gpu else False,
            **extra,
        )
    return MPSBackend, MPSConfig(
        dt=dt,
        observables=observables,
        log_level=log_level,
        with_modulation=with_modulation,
        num_gpus_to_use=None if use_gpu else 0,
        max_bond_dim=max_bond_dim,
        optimize_qubit_ordering=True,
        **extra,
    )


def emulate(
    sequence: pulser.Sequence,
    evaluation_times_ns: list[int],
    shots: int = 1000,
    exact: bool = False,
    use_gpu: bool = True,
    verbose: bool = False,
    with_modulation: bool = False,
) -> dict:
    """Run ``sequence`` and sample ``shots`` bitstrings at each evaluation time.

    Returns ``{"counts": {t_ns: {bitstring: count}}}``; with ``exact=True`` (emu-sv
    only) also ``"occupation"`` and ``"correlation"`` (``<n_i n_j>``) per time, as
    exact state-vector expectations.
    """
    duration = sequence.get_duration()
    if max(evaluation_times_ns) > duration or min(evaluation_times_ns) <= 0:
        raise ValueError("evaluation times must lie in (0, sequence duration]")
    rel = [float(t) / duration for t in evaluation_times_ns]
    observables = [BitStrings(evaluation_times=rel, num_shots=shots)]
    if exact:
        observables += [Occupation(evaluation_times=rel), CorrelationMatrix(evaluation_times=rel)]
    n = len(sequence.register.qubit_ids)
    backend_cls, config = make_config(n, observables, use_gpu, verbose, with_modulation)
    if exact and backend_cls is not SVBackend:
        raise ValueError("exact expectations are only provided with the state-vector backend")
    results = backend_cls(sequence, config=config).run()

    out: dict = {"counts": {}, "occupation": {}, "correlation": {}}
    keys = {"counts": observables[0]}
    if exact:
        keys |= {"occupation": observables[1], "correlation": observables[2]}
    for key, obs in keys.items():
        for t_rel in results.get_result_times(obs):
            t_ns = round(duration * t_rel)
            value = results.get_result(obs, t_rel)
            out[key][t_ns] = dict(value) if key == "counts" else _real(value)
    return out if exact else {"counts": out["counts"]}


def _real(value):
    import numpy as np

    arr = np.asarray(value.cpu() if hasattr(value, "cpu") else value)
    if np.iscomplexobj(arr):
        if not np.allclose(arr.imag, 0, atol=1e-10):
            raise ValueError("observable has a non-zero imaginary part")
        arr = arr.real
    return arr.astype(float)


def emulate_expectations(
    sequence: pulser.Sequence,
    evaluation_times_ns,
    use_gpu: bool = True,
    verbose: bool = False,
    **config,
) -> tuple:
    """Occupation ``<n_i>`` and connected correlation ``<n_i n_j> - <n_i><n_j>``, no shots.

    Returns ``(occupation [T, N], connected [T, N, N])``, time-major, from the state of
    either backend (emu-sv exact; emu-mps within its truncation). ``config`` is passed to
    :func:`make_config` (``dt``, ``sv_max_qubits``, ``max_bond_dim``, ``interaction_matrix``).
    """
    import numpy as np

    duration = sequence.get_duration()
    rel = [float(t) / duration for t in evaluation_times_ns]
    occ_obs, corr_obs = Occupation(evaluation_times=rel), CorrelationMatrix(evaluation_times=rel)
    n = len(sequence.register.qubit_ids)
    backend_cls, cfg = make_config(n, [occ_obs, corr_obs], use_gpu, verbose, **config)
    results = backend_cls(sequence, config=cfg).run()
    times = results.get_result_times(occ_obs)
    occ = np.stack([_real(results.get_result(occ_obs, t)) for t in times])
    corr = np.stack([_real(results.get_result(corr_obs, t)) for t in times])
    return occ, corr - occ[:, :, None] * occ[:, None, :]
