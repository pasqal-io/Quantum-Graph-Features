"""Placement sensitivity: emulate AFM correlations under Gaussian position noise.

Four ZINC*_subset test graphs (sizes closest to 8, 12, 16, 20 nodes). For each, the
reference register (stored ``pos``) and 10 noisy copies per noise level
``sigma / d_min`` in {0.02, 0.05, 0.1, 0.2, 0.35, 0.5} (``d_min`` = native minimum
distance; registers whose minimum distance drops below r_NN = 5 um are rescaled).
Each case is emulated (AFM quench, 1000 shots at the 20 times) and its correlation
tensor ``C [T, N, N]`` saved as ``<out>/afm/g<idx>_s<sigma>_r<run>_noise.npz``, the
format read by ``analyze.py``.

Note: the stored study was emulated with pulse modulation (finite rise time of the
hardware channel), unlike the released features. That needs the private channel
modulation bandwidth, so only the unmodulated (feature-pipeline) setting is available.

Re-embedding (``--reembed``): alternative valid layouts of the same graph from Blade
(``qoolqit.embedding.Blade``, QUBO = adjacency with -1 diagonal, 5D -> 2D, random seeded
starts, rescaled to r_NN). ``reembed``: the first 10 candidates; ``post_selected_embedding``:
the 10 best of 100 candidates by rank sum of ELU (low) and GR (high). The stored study
used the internal force-based embedder instead, so its numbers are not expected to match.

    uv run --extra generate python experiments/layout_sensitivity/run.py --graph-idx 4 \
        --out results/layout_sensitivity
"""

import argparse
from pathlib import Path

import numpy as np

from qgf.data import load
from qgf.generate.build import raw_correlations
from qgf.generate.emulate import emulate
from qgf.generate.raw import RawResults, encode_counts
from qgf.generate.sequences import EVALUATION_TIMES_NS, quench_sequence, register_from_positions

R_NN = 5.0
SIGMA_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
TARGET_SIZES = (8, 12, 16, 20)
N_REALIZATIONS = 10


def select_graph_indices(sizes, target_sizes=TARGET_SIZES) -> list[int]:
    """Closest graph to each target size, distinct, lowest index on ties."""
    sizes = np.asarray(sizes)
    chosen: list[int] = []
    for target in target_sizes:
        for idx in np.argsort(np.abs(sizes - target), kind="stable"):
            if int(idx) not in chosen:
                chosen.append(int(idx))
                break
    return sorted(chosen)


def min_distance(pos: np.ndarray) -> float:
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return float(d[np.triu_indices(len(pos), 1)].min())


def perturb_positions(pos, sigma_rel: float, rng, d_min: float) -> tuple[np.ndarray, bool]:
    """Isotropic Gaussian noise of std ``sigma_rel * d_min``; rescale to r_NN if needed."""
    noisy = np.asarray(pos, float) + rng.normal(scale=sigma_rel * d_min, size=np.shape(pos))
    if min_distance(noisy) < R_NN:
        return noisy * (R_NN / min_distance(noisy)), True
    return noisy, False


def case_seed(graph_idx: int, sigma_rel: float, realization: int, mode: str = "noise") -> int:
    offset = {"noise": 0, "reembed": 1, "post_selected": 2}.get(mode, 0)
    return (graph_idx * 100003 + round(sigma_rel * 1000) * 997 + realization * 31 + offset * 7) % (
        2**31
    )


def edge_length_uniformity(pos: np.ndarray, edges: np.ndarray) -> float:
    """ELU = Var(r) / mean(r)^2 over edges, r = d / min_E d (lower is better)."""
    d = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    r = d / d.min()
    return float(np.var(r) / np.mean(r) ** 2)


def gap_ratio(pos: np.ndarray, edges: np.ndarray, q: float = 0.005) -> float:
    """Mean of the q-quantile tail of non-edge distances / longest edge (higher is better)."""
    n = len(pos)
    iu = np.triu_indices(n, k=1)
    d_all = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    edge_set = {frozenset(map(int, e)) for e in edges}
    is_edge = np.array([frozenset((int(i), int(j))) in edge_set for i, j in zip(*iu, strict=True)])
    d_edge, d_non = d_all[is_edge], d_all[~is_edge]
    if d_non.size == 0 or d_edge.size == 0:
        return float("nan")
    tail = d_non[d_non <= np.quantile(d_non, q)]
    return float(tail.mean() / d_edge.max())


BLADE_DIMENSIONS = (5, 4, 3, 2)  # (3, 2) gave no unit-disk-like layout for N >= 11
BLADE_STEPS = 200


def blade_positions(adj: np.ndarray, seed: int) -> np.ndarray:
    """One Blade embedding of the graph (random start from ``seed``), rescaled to r_NN."""
    import warnings

    from qoolqit.embedding import Blade, BladeConfig

    n = len(adj)
    qubo = adj.copy()
    np.fill_diagonal(qubo, -1.0)
    start = np.random.default_rng(seed).uniform(size=(n, BLADE_DIMENSIONS[0]))
    cfg = BladeConfig(
        dimensions=BLADE_DIMENSIONS, steps_per_round=BLADE_STEPS, starting_positions=start
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = Blade(cfg).embed(qubo).coords
    pos = np.array([coords[i] for i in range(n)], dtype=float)
    return pos * (R_NN / min_distance(pos))


def blade_candidates(adj: np.ndarray, edges: np.ndarray, n: int, base_seed: int) -> list[dict]:
    out = []
    for k in range(n):
        pos = blade_positions(adj, base_seed + k)
        out.append(
            {
                "positions": pos,
                "elu": edge_length_uniformity(pos, edges),
                "gr": gap_ratio(pos, edges),
                "seed": base_seed + k,
            }
        )
    return out


def select_best(scored: list[dict], k: int) -> list[dict]:
    """Top-k by rank sum: ELU ascending + GR descending."""
    from scipy.stats import rankdata

    valid = [s for s in scored if np.isfinite(s["elu"]) and np.isfinite(s["gr"])]
    rank = rankdata([s["elu"] for s in valid]) + rankdata([-s["gr"] for s in valid])
    return [valid[i] for i in np.argsort(rank, kind="stable")[:k]]


def correlations(positions, use_gpu: bool) -> np.ndarray:
    seq = quench_sequence(register_from_positions(positions), "afm")
    counts = emulate(seq, EVALUATION_TIMES_NS, use_gpu=use_gpu)["counts"]
    enc = [encode_counts(counts[t]) for t in EVALUATION_TIMES_NS]
    offsets = np.cumsum([0] + [len(b) for b, _ in enc])
    raw = RawResults(
        initial_id=np.zeros(1, int),
        num_nodes=np.array([len(positions)]),
        times_ns=np.array(EVALUATION_TIMES_NS),
        offsets=offsets,
        bits=np.concatenate([b for b, _ in enc]),
        shot_counts=np.concatenate([c for _, c in enc]),
    )
    return raw_correlations(raw, 0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--graph-idx", type=int, default=None, help="index in zinc_afm test")
    parser.add_argument("--list-graphs", action="store_true")
    parser.add_argument("--sigmas", type=float, nargs="*", default=list(SIGMA_GRID))
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reembed", action="store_true", help="Blade re-embeddings instead")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/layout_sensitivity"))
    args = parser.parse_args(argv)

    graphs, _ = load("zinc_afm", "test")
    if args.list_graphs or args.graph_idx is None:
        idx = select_graph_indices([int(g.num_nodes) for g in graphs])
        print(" ".join(map(str, idx)), "| sizes", [int(graphs[i].num_nodes) for i in idx])
        return
    data = graphs[args.graph_idx]
    ref = data.pos.double().numpy()
    d_min = min_distance(ref)
    save = {
        "times": np.array(EVALUATION_TIMES_NS),
        "graph_idx": args.graph_idx,
        "num_nodes": int(data.num_nodes),
        "pulse": "afm",
        "d_min": d_min,
    }
    if args.reembed:
        n = int(data.num_nodes)
        edges = data.edge_index.numpy().T
        adj = np.zeros((n, n))
        adj[edges[:, 0], edges[:, 1]] = adj[edges[:, 1], edges[:, 0]] = 1.0
        base = case_seed(args.graph_idx, 0.0, 0, "post_selected")
        cands = blade_candidates(adj, edges, args.candidates, base)
        sets = {
            "reembed": cands[: args.realizations],
            "post_selected_embedding": select_best(cands, args.realizations),
        }
        for mode, chosen in sets.items():
            for r, c in enumerate(chosen):
                path = args.out / "afm" / f"g{args.graph_idx}_s0.000_r{r}_{mode}.npz"
                corr = correlations(c["positions"], not args.cpu)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    corr=corr,
                    positions=c["positions"],
                    sigma_rel=0.0,
                    realization=r,
                    mode=mode,
                    rescaled=False,
                    elu=c["elu"],
                    gr=c["gr"],
                    seed=c["seed"],
                    **save,
                )
                print(f"wrote {path} (ELU={c['elu']:.4g}, GR={c['gr']:.3g})", flush=True)
        return
    for sigma in args.sigmas:
        for r in range(1 if sigma == 0.0 else args.realizations):
            path = args.out / "afm" / f"g{args.graph_idx}_s{sigma:.3f}_r{r}_noise.npz"
            if path.exists():
                continue
            rescaled = False
            pos = ref
            if sigma > 0:
                rng = np.random.default_rng(case_seed(args.graph_idx, sigma, r))
                pos, rescaled = perturb_positions(ref, sigma, rng, d_min)
            corr = correlations(pos, use_gpu=not args.cpu)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                corr=corr,
                times=np.array(EVALUATION_TIMES_NS),
                positions=pos,
                sigma_rel=sigma,
                graph_idx=args.graph_idx,
                num_nodes=int(data.num_nodes),
                realization=r,
                pulse="afm",
                mode="noise",
                d_min=d_min,
                rescaled=rescaled,
            )
            print(f"wrote {path} (rescaled={rescaled})", flush=True)


if __name__ == "__main__":
    main()
