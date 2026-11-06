"""One trial of the downstream GNN benchmark (port of the internal ``train_mutag.py`` /
``train_zinc_small.py`` of the PE dataloader template).

Data, folds and encodings are shared with the GT port (:mod:`qgf.downstream.gt`): MUTAG
10-fold CV stratified by (label, size quartile), fold ``k`` test, fold ``k+1`` validation;
ZINC*_subset fixed split; every encoding block z-scored with training-split statistics.

Training (App. E.2, same for every backbone and feature set): AdamW, linear warmup (start
factor 1e-6) then ReduceLROnPlateau (factor 0.5, patience 50), early stopping on the
validation split (MUTAG: macro-F1, ZINC: MAE), test metrics at the best validation epoch.
Model: size ``large`` (4 x 64 message-passing layers, 4 x 64 MLP head), mean pooling,
dropout 0.1, LayerNorm, one attention head; GCN shifts its edge weights per graph.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from qgf.data import DEFAULT_ROOT
from qgf.downstream.gnns.model import ALL_MODELS, EDGE_WEIGHT_MODELS, GNNWithPE
from qgf.downstream.gt import data as gt_data
from qgf.downstream.gt.benchmark import build_splits, transform_splits
from qgf.downstream.gt.pe_transforms import CONFIGS

MUTAG_SEEDS = [1, 2, 3, 4, 5]
ZINC_SEEDS = [0, 1, 2, 3, 4]

MODEL_SIZES: dict[str, dict[str, list[int]]] = {
    "small": {"gnn_layers": [32, 32], "mlp_layers": [32, 32]},
    "medium": {"gnn_layers": [32, 32, 32], "mlp_layers": [32, 32, 32]},
    "large": {"gnn_layers": [64] * 4, "mlp_layers": [64] * 4},
    "huge": {"gnn_layers": [128] * 4, "mlp_layers": [128] * 4},
    "very_huge": {"gnn_layers": [256] * 5, "mlp_layers": [256] * 3},
}

HPARAMS: dict[str, dict[str, Any]] = {
    "mutag": {
        "max_epochs": 300,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "patience": 75,
        "warmup_epochs": 20,
    },
    "zinc": {
        "max_epochs": 1000,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "patience": 200,
        "warmup_epochs": 50,
    },
}
MODEL_KWARGS: dict[str, Any] = {"pooling": "mean", "dropout": 0.1, "num_heads": 1}
PLATEAU_PATIENCE = 50


def make_loader(graphs: list[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=gt_data.custom_collate_with_pe,
    )


def build_model(
    dataset: str, model_type: str, abs_dim: int, rel_dim: int, size: str = "large"
) -> GNNWithPE:
    vocab = {
        "mutag": (gt_data.MUTAG_NUM_ATOM_TYPES, gt_data.MUTAG_NUM_BOND_TYPES, 2),
        "zinc": (gt_data.ZINC_NUM_ATOM_TYPES, gt_data.ZINC_NUM_BOND_TYPES, 1),
    }[dataset]
    atoms, bonds, out = vocab
    return GNNWithPE(
        model_type,
        num_atom_types=atoms,
        num_bond_types=bonds,
        abs_pe_dim=abs_dim,
        rel_pe_dim=rel_dim,
        output_dim=out,
        edge_weight_shift=model_type in EDGE_WEIGHT_MODELS,
        **MODEL_SIZES[size],
        **MODEL_KWARGS,
    )


def _predict(model: GNNWithPE, loader: DataLoader, device: torch.device, classify: bool):
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)  # [B, 2] (MUTAG) or [B, 1] (ZINC)
        preds.append(out if classify else out.squeeze(-1))
        targets.append(batch.y if classify else batch.y.float())
    return torch.cat(preds), torch.cat(targets)


@torch.no_grad()
def evaluate(model: GNNWithPE, loader: DataLoader, device: torch.device, classify: bool) -> dict:
    """``{"accuracy", "f1_macro"}`` (MUTAG) or ``{"mae", "rmse"}`` (ZINC)."""
    model.eval()
    preds, targets = _predict(model, loader, device, classify)
    if classify:
        pred = preds.argmax(dim=-1).cpu().numpy()
        true = targets.squeeze().long().cpu().numpy()
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "f1_macro": float(f1_score(true, pred, average="macro")),
        }
    return {
        "mae": F.l1_loss(preds, targets).item(),
        "rmse": torch.sqrt(F.mse_loss(preds, targets)).item(),
    }


def train_epoch(model, loader, optimizer, device, scaler, classify: bool) -> None:
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        amp = torch.amp.autocast("cuda") if scaler is not None else contextlib.nullcontext()
        with amp:
            out = model(batch)
            if classify:
                loss = F.cross_entropy(out, batch.y.squeeze().long())
            else:
                loss = F.l1_loss(out.squeeze(-1), batch.y.float())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def fit(
    model: GNNWithPE,
    splits: dict[str, list[Data]],
    classify: bool,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    warmup_epochs: int,
    amp: bool = False,
) -> dict[str, Any]:
    """Train with early stopping on the validation split; test metrics at the best epoch."""
    device = next(model.parameters()).device
    loaders = {
        k: make_loader(splits[k], batch_size, shuffle=(k == "train"))
        for k in ("train", "val", "test")
    }
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup = (
        LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
        if warmup_epochs > 0
        else None
    )
    plateau = ReduceLROnPlateau(
        optimizer, mode="max" if classify else "min", factor=0.5, patience=PLATEAU_PATIENCE
    )
    # The published runs trained on CPU in float32; GPU runs keep float32 unless amp=True
    # (the original script's mixed precision on GPU).
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None
    key = "f1_macro" if classify else "mae"
    sign = 1.0 if classify else -1.0  # maximise F1, minimise MAE

    best_val, best_state, best_epoch, waited, epoch = -float("inf"), None, 0, 0, 0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, loaders["train"], optimizer, device, scaler, classify)
        val = evaluate(model, loaders["val"], device, classify)[key]
        if warmup is not None and epoch <= warmup_epochs:
            warmup.step()
        else:
            plateau.step(val)
        if sign * val > best_val:
            best_val, best_epoch, waited = sign * val, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device, classify)
    return {
        "best_epoch": best_epoch,
        f"best_val_{key}": sign * best_val,
        "epochs_run": epoch,
        **{f"test_{k}": v for k, v in test.items()},
    }


def run_trial(
    dataset: str,
    model_type: str,
    config: str,
    seed: int,
    fold: int | None = None,
    output_dir: Path | None = None,
    root: Path = DEFAULT_ROOT,
    size: str = "large",
    max_epochs: int | None = None,
    graphs: dict[str, list[Data]] | list[Data] | None = None,
    subset: str | None = None,
) -> dict[str, Any]:
    """Run one (backbone, config, fold, seed) trial; write ``results.json`` in ``output_dir``.

    ``max_epochs`` overrides the paper value (smoke tests); ``graphs`` passes an in-memory
    dataset (MUTAG: list, ZINC: split dict) instead of the released one; ``subset``
    restricts MUTAG (``"exact173"``: the graphs with exact features).
    """
    if model_type not in ALL_MODELS:
        raise ValueError(f"unknown backbone {model_type!r}")
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}")
    classify = dataset == "mutag"
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw = build_splits(dataset, config, fold, root, graphs, subset)
    splits, abs_dim, rel_dim = transform_splits(config, raw)

    torch.manual_seed(seed)  # re-seed before model initialisation, as in the original
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dataset, model_type, abs_dim, rel_dim, size).to(device)
    hp = dict(HPARAMS[dataset])
    if max_epochs is not None:
        hp["max_epochs"] = max_epochs
    metrics = fit(model, splits, classify, **hp)

    results: dict[str, Any] = {
        "dataset": dataset,
        "model_type": model_type,
        "config": config,
        "subset": subset,
        "size": size,
        "fold": fold,
        "seed": seed,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "abs_pe_dim": abs_dim,
        "rel_pe_dim": rel_dim,
        **metrics,
        "training_config": hp,
        "model_config": {**MODEL_SIZES[size], **MODEL_KWARGS},
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
