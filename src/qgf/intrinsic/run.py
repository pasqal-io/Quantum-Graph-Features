"""Run intrinsic-benchmark probes and write ``result.json`` files.

Ported from the original internal implementation. Raw classical views and targets are
computed once per dataset and cached (pickle) under ``--cache_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from qgf.intrinsic.data_loading import DATASETS, load_dataset
from qgf.intrinsic.deepset_probe import evaluate_deepset_cv, evaluate_deepset_graph_cv
from qgf.intrinsic.metrics import BENCHMARK_REGISTRY, TASKS, compute_targets, get_category
from qgf.intrinsic.pe_construction import (
    PE_GROUPS,
    PE_TYPES,
    attach,
    build_edge_pe,
    build_graph_pe,
    build_node_pe,
    classical_views,
    data_to_graph,
    normalize_dataset,
)

DEFAULT_CACHE_DIR = "results/intrinsic/cache"
_BUILDERS = {
    "node_prediction": build_node_pe,
    "edge_prediction": build_edge_pe,
    "graph_prediction": build_graph_pe,
}


def available_pe_types(dataset_name: str) -> list[str]:
    """PE types of the paper benchmark for a dataset (HW only on MUTAG, residuals only on ZINC)."""
    pes = []
    for pe, groups in PE_GROUPS.items():
        if "hw_afm" in groups and dataset_name != "MUTAG":
            continue
        if {"fmres", "afmres"} & set(groups) and dataset_name != "ZINC_small":
            continue
        if pe.startswith("RRWP_Laplacian_Ring_Geometric") and dataset_name != "ZINC_small":
            continue  # the Geometric hybrids were only run on ZINC
        pes.append(pe)
    return pes


def _cached(path: Path, compute):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return value


def _pool_map(func, items, workers: int):
    if workers <= 1:
        return [func(x) for x in items]
    with Pool(workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        return pool.map(func, items, chunksize=max(1, len(items) // (workers * 8)))


def _task_targets(args):
    benchmark, graph = args
    return compute_targets(benchmark, [graph])


def _views_job(args):
    edge_index, num_nodes, pos = args
    d = Data(
        edge_index=torch.from_numpy(edge_index), num_nodes=num_nodes, pos=torch.from_numpy(pos)
    )
    return classical_views(d)


def get_targets(dataset_name, benchmark, dataset, cache_dir, workers=1):
    def compute():
        graphs = [data_to_graph(d) for d in dataset]
        parts = _pool_map(_task_targets, [(benchmark, g) for g in graphs], workers)
        if get_category(benchmark) == "graph_prediction":
            return np.concatenate(parts)
        return [p[0] for p in parts]

    return _cached(Path(cache_dir) / dataset_name / "targets" / f"{benchmark}.pkl", compute)


def prepare_dataset(dataset_name, cache_dir=DEFAULT_CACHE_DIR, workers=1):
    """Raw (unnormalised) dataset: quantum features + classical views."""
    dataset = load_dataset(dataset_name)
    views = _cached(
        Path(cache_dir) / dataset_name / "classical_views.pkl",
        lambda: _pool_map(
            _views_job,
            [(d.edge_index.numpy(), d.num_nodes, d.pos.numpy()) for d in dataset],
            workers,
        ),
    )
    attach(dataset, views)
    return dataset


def run_benchmark(
    dataset_name: str,
    pe_type: str,
    benchmark: str,
    output_dir: str,
    seed: int = 42,
    cache_dir: str = DEFAULT_CACHE_DIR,
    dataset=None,
) -> dict:
    """Run one (dataset, pe_type, task) probe and write ``output_dir/result.json``."""
    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    category = get_category(benchmark)
    classification = BENCHMARK_REGISTRY[category][benchmark]["classification"]
    if dataset is None:
        dataset = prepare_dataset(dataset_name, cache_dir)
    targets = get_targets(dataset_name, benchmark, dataset, cache_dir)
    groups = PE_GROUPS[pe_type]
    build = _BUILDERS[category]

    def pe_for(fit_idx=None):
        return [build(d, pe_type) for d in normalize_dataset(dataset, groups, fit_idx)]

    # Normalisation fitted on the training graphs of each fold, except for random
    # features (nothing to normalise).
    pe = pe_for if groups else pe_for()

    if category == "graph_prediction":
        raw, std, folds = evaluate_deepset_graph_cv(pe, targets, seed=seed)
    else:
        raw, std, folds = evaluate_deepset_cv(pe, targets, classification=classification, seed=seed)

    result = {
        "dataset": dataset_name,
        "pe_type": pe_type,
        "benchmark": benchmark,
        "category": category,
        "metric": "accuracy" if classification else "mae",
        "raw_value": raw,
        "std_dev": std,
        "fold_scores": folds,
        "seed": seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--pe_type", nargs="+", default=["all"], help="PE types or 'all'")
    parser.add_argument("--benchmark", nargs="+", default=["all"], help="tasks or 'all'")
    parser.add_argument("--output_dir", required=True, help="writes <dir>/<ds>/<pe>/<task>/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=1, help="processes for target/view caching")
    parser.add_argument("--prepare_only", action="store_true", help="only fill the cache")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args(argv)

    pes = available_pe_types(args.dataset) if args.pe_type == ["all"] else args.pe_type
    tasks = TASKS if args.benchmark == ["all"] else args.benchmark
    for pe in pes:
        if pe not in PE_TYPES:
            parser.error(f"unknown pe_type {pe}")

    dataset = prepare_dataset(args.dataset, args.cache_dir, args.workers)
    for task in tasks:
        get_targets(args.dataset, task, dataset, args.cache_dir, args.workers)
    if args.prepare_only:
        return
    for pe in pes:
        for task in tasks:
            out = os.path.join(args.output_dir, args.dataset, pe, task)
            if args.skip_existing and os.path.exists(os.path.join(out, "result.json")):
                continue
            r = run_benchmark(args.dataset, pe, task, out, args.seed, args.cache_dir, dataset)
            print(
                f"{args.dataset} {pe} {task}: {r['raw_value']:.6g} ± {r['std_dev']:.3g} "
                f"({r['seconds']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
