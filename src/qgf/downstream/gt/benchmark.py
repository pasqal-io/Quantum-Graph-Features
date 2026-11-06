"""One trial of the downstream GT benchmark (port of the internal run_pe_benchmark.py).

- MUTAG: 10-fold CV stratified by (label, graph-size quartile), ``random_state=42``,
  identical folds for every config. Fold ``k`` is the test set, fold ``(k+1) % 10`` the
  validation set (early stopping), the 8 others the training set. Seeds 1..5.
- ZINC*_subset: fixed train/val/test split (9475/946/947), seeds 0..9.

Hyperparameters are those of Table 6 of the paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

from qgf.data import DEFAULT_ROOT
from qgf.downstream.gt import data as gt_data
from qgf.downstream.gt.model.config import ModelConfig
from qgf.downstream.gt.pe_transforms import CONFIGS, PEBenchmarkTransform, quantum_sources
from qgf.downstream.gt.train import train_simple

N_FOLDS = 10
FOLD_RANDOM_STATE = 42
MUTAG_SEEDS = [1, 2, 3, 4, 5]
ZINC_SEEDS = list(range(10))
RRWP_K = 20
RANDOM_DIM = 20

HPARAMS: dict[str, dict[str, Any]] = {
    "mutag": {
        "model": {
            "hidden_dim": 32,
            "num_layers": 3,
            "num_heads": 4,
            "ffn_dim": 64,
            "dropout": 0.3,
            "pooling": "mean",
            "num_atom_types": gt_data.MUTAG_NUM_ATOM_TYPES,
            "num_bond_types": gt_data.MUTAG_NUM_BOND_TYPES,
            "task_type": "classification",
            "num_classes": 2,
        },
        "train": {
            "max_epochs": 300,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,
            "patience": 75,
            "warmup_epochs": 20,
            "loss_fn": "cross_entropy",
        },
    },
    "zinc": {
        "model": {"hidden_dim": 256, "num_heads": 16, "ffn_dim": 512},  # num_layers=4, dropout=0.1
        "train": {
            "max_epochs": 1000,
            "batch_size": 128,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "patience": 200,
            "warmup_epochs": 50,
            "loss_fn": "l1",
        },
    },
}


def make_kfold_indices(
    data_list: list[Data], n_folds: int = N_FOLDS, random_state: int = FOLD_RANDOM_STATE
) -> list[list[int]]:
    """Stratified fold test-index lists; key = label * 4 + graph-size quartile."""
    labels = np.array([int(d.y.item()) for d in data_list])
    sizes = np.array([d.x.shape[0] for d in data_list])
    size_buckets = np.digitize(sizes, np.percentile(sizes, [25, 50, 75]))  # 0..3
    strat = labels * 4 + size_buckets
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [test.tolist() for _, test in skf.split(np.zeros(len(data_list)), strat)]


def mutag_split(graphs: list[Data], fold: int) -> dict[str, list[Data]]:
    folds = make_kfold_indices(graphs)
    test_idx, val_idx = folds[fold], folds[(fold + 1) % N_FOLDS]
    held = set(test_idx) | set(val_idx)
    train_idx = [i for i in range(len(graphs)) if i not in held]
    return {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }


def build_splits(
    dataset: str,
    config: str,
    fold: int | None,
    root: Path = DEFAULT_ROOT,
    graphs: dict[str, list[Data]] | list[Data] | None = None,
    subset: str | None = None,
) -> dict[str, list[Data]]:
    """Raw train/val/test graphs with the config's quantum sources attached.

    ``subset`` restricts MUTAG (:data:`qgf.downstream.gt.data.MUTAG_SUBSETS`); the folds
    are then built on the subset.
    """
    sources = quantum_sources(config)
    if dataset == "mutag":
        if fold is None:
            raise ValueError("MUTAG needs a fold")
        return mutag_split(
            graphs if graphs is not None else gt_data.load_mutag(sources, root, subset), fold
        )
    if subset is not None:
        raise ValueError("subsets are defined for MUTAG only")
    if dataset == "zinc":
        return graphs if graphs is not None else gt_data.load_zinc(sources, root)
    raise ValueError(f"unknown dataset {dataset!r}")


def transform_splits(
    config: str, raw: dict[str, list[Data]]
) -> tuple[dict[str, list[Data]], int, int]:
    """Apply the config's encodings (stats fitted on train); return splits and PE dims."""
    transform = PEBenchmarkTransform(config, raw["train"], rrwp_k=RRWP_K, random_dim=RANDOM_DIM)
    out = {k: [transform(d) for d in raw[k]] for k in ("train", "val", "test")}
    sample = out["train"][0]
    ob = getattr(sample, "onebody_features", None)
    tb = getattr(sample, "twobody_features", None)
    return out, (0 if ob is None else ob.shape[1]), (0 if tb is None else tb.shape[2])


def run_trial(
    dataset: str,
    config: str,
    seed: int,
    fold: int | None = None,
    output_dir: Path | None = None,
    root: Path = DEFAULT_ROOT,
    max_epochs: int | None = None,
    graphs: dict[str, list[Data]] | list[Data] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one (config, fold, seed) trial and write ``results.json`` in ``output_dir``.

    ``max_epochs`` overrides the paper value (smoke tests only). ``graphs`` lets tests
    pass a tiny in-memory dataset (MUTAG: list, ZINC: split dict). ``model_overrides``
    replaces model hyperparameters (capacity study, Fig. 6).
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}")
    hp = {**HPARAMS[dataset], "model": {**HPARAMS[dataset]["model"], **(model_overrides or {})}}
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw = build_splits(dataset, config, fold, root, graphs)
    splits, abs_dim, rel_dim = transform_splits(config, raw)
    model_config = ModelConfig(abs_pe_dim=abs_dim, rel_pe_dim=rel_dim, **hp["model"])

    # Re-seed just before training (as in the original runner)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_kw = dict(hp["train"])
    if max_epochs is not None:
        train_kw["max_epochs"] = max_epochs
    test_metrics, model, info = train_simple(
        splits["train"], splits["val"], splits["test"], model_config, **train_kw
    )

    results: dict[str, Any] = {
        "dataset": dataset,
        "config": config,
        "fold": fold,
        "seed": seed,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "abs_pe_dim": abs_dim,
        "rel_pe_dim": rel_dim,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **info,
        "training_config": train_kw,
        "model_config": hp["model"],
    }
    if dataset == "mutag":
        results.update(test_accuracy=test_metrics.accuracy, test_f1_macro=test_metrics.f1_macro)
    else:
        results.update(test_mae=test_metrics.mae, test_rmse=test_metrics.rmse)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
