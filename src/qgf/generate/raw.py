"""Raw measurement records: bitstring counts per graph and per measurement time.

On-disk format (one ``.npz`` per dataset, written by :func:`save_raw`):

- ``initial_id`` int64 [G]: graph id, same as ``initial_id`` in the processed datasets.
- ``num_nodes`` int64 [G]: atoms per register.
- ``times_ns`` int64 [T]: measurement times (the time keys of the original records).
- ``offsets`` int64 [G*T + 1]: record ``(g, t)`` (graph-major, ``r = g*T + t``) owns
  ``bits[offsets[r]:offsets[r+1]]`` and ``counts[...]`` (empty for a missing record).
- ``bits`` uint64 [K]: unique bitstrings, bit ``i`` (LSB first) = atom ``i`` in Rydberg state.
- ``counts`` uint32 [K]: number of shots for each bitstring.
- ``meta`` str: JSON metadata.

Only numpy is needed to read it. :func:`load_raw` returns a :class:`RawResults`
whose ``counts(g, t)`` gives back pulser-style ``{bitstring: count}`` dicts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

# The four QPU runs of ``mutag_hw_afm`` (time index order T2, T2.5, T4, T15); T4 is Fig. 5.
QPU_RUNS = (
    "mutag_qpu_afm_t2_305ns",
    "mutag_qpu_afm_t2p5_512ns",
    "mutag_qpu_afm_t4_715ns",
    "mutag_qpu_afm_t15_4000ns",
)


def raw_path(raw_dir: str | Path, name: str) -> Path:
    """Released raw file of dataset ``name`` under ``raw_dir`` (``data/raw``).

    Layout: ``emulation/mutag/{afm,fm}.npz``, ``emulation/zinc/{afm,fm}_<split>.npz``,
    ``embeddings/zinc_<split>.npz`` (register positions), ``qpu/mutag_afm/<run>.npz``
    (e.g. ``mutag_qpu_afm_t4_715ns`` -> ``qpu/mutag_afm/t4_715ns.npz``),
    ``layout_sensitivity/...``. Names without a known prefix stay flat (``<name>.npz``).
    """
    raw_dir = Path(raw_dir)
    for prefix, sub in (
        ("mutag_qpu_afm_", "qpu/mutag_afm/"),
        ("zinc_embeddings_", "embeddings/zinc_"),
        ("mutag_", "emulation/mutag/"),
        ("zinc_", "emulation/zinc/"),
    ):
        if name.startswith(prefix):
            return raw_dir / f"{sub}{name.removeprefix(prefix)}.npz"
    return raw_dir / f"{name}.npz"


def encode_counts(counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """``{"0110": 3, ...}`` -> ``(bits uint64 [K], counts uint32 [K])``."""
    keys = list(counts)
    if keys and len(keys[0]) > 64:
        raise ValueError("registers above 64 atoms are not supported")
    bits = np.array([int(s[::-1], 2) for s in keys], dtype=np.uint64)
    return bits, np.array([counts[k] for k in keys], dtype=np.uint32)


def decode_counts(bits: np.ndarray, counts: np.ndarray, num_nodes: int) -> dict[str, int]:
    return {
        format(int(b), f"0{num_nodes}b")[::-1]: int(c) for b, c in zip(bits, counts, strict=True)
    }


@dataclass
class RawResults:
    initial_id: np.ndarray
    num_nodes: np.ndarray
    times_ns: np.ndarray
    offsets: np.ndarray
    bits: np.ndarray
    shot_counts: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.initial_id)

    def _slice(self, g: int, t: int) -> slice:
        r = g * len(self.times_ns) + t
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))

    def counts(self, g: int, t: int) -> dict[str, int]:
        """Bitstring counts of graph index ``g`` at time index ``t`` ({} if missing)."""
        s = self._slice(g, t)
        return decode_counts(self.bits[s], self.shot_counts[s], int(self.num_nodes[g]))

    def bit_matrix(self, g: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        """``(bits [K, N] float, weights [K] float)`` without going through strings."""
        s = self._slice(g, t)
        n = int(self.num_nodes[g])
        b = self.bits[s][:, None] >> np.arange(n, dtype=np.uint64)[None, :]
        return (b & np.uint64(1)).astype(float), self.shot_counts[s].astype(float)

    def index_of(self, initial_id: int) -> int:
        return int(np.flatnonzero(self.initial_id == initial_id)[0])


def save_raw(
    path: str | Path,
    records: list[dict[int, dict[str, int]]],
    initial_id: list[int],
    num_nodes: list[int],
    times_ns: list[int],
    meta: dict | None = None,
) -> None:
    """Write ``records[g] = {time_ns: {bitstring: count}}`` (missing times allowed)."""
    if not (len(records) == len(initial_id) == len(num_nodes)):
        raise ValueError("records, initial_id and num_nodes must have the same length")
    bits, counts, offsets = [], [], [0]
    for rec in records:
        for t in times_ns:
            b, c = encode_counts(rec.get(t, {}))
            bits.append(b)
            counts.append(c)
            offsets.append(offsets[-1] + len(b))
    meta = {"format_version": FORMAT_VERSION, **(meta or {})}
    np.savez_compressed(
        path,
        initial_id=np.asarray(initial_id, dtype=np.int64),
        num_nodes=np.asarray(num_nodes, dtype=np.int64),
        times_ns=np.asarray(times_ns, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        bits=np.concatenate(bits) if bits else np.zeros(0, np.uint64),
        counts=np.concatenate(counts) if counts else np.zeros(0, np.uint32),
        meta=np.array(json.dumps(meta)),
    )


def load_raw(path: str | Path) -> RawResults:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported raw format {meta.get('format_version')}")
        return RawResults(
            initial_id=z["initial_id"],
            num_nodes=z["num_nodes"],
            times_ns=z["times_ns"],
            offsets=z["offsets"],
            bits=z["bits"],
            shot_counts=z["counts"],
            meta=meta,
        )
