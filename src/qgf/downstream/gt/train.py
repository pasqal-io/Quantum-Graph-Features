"""Training loop of the downstream GT benchmark (port of the original internal implementation).

Only ``train_simple`` and its helpers are kept. Changes from the original:
TensorBoard logging and the PE diagnostics printout are removed (logging only);
everything that touches the RNG (loader creation, the first ``next(iter(train_loader))``
draw, model initialisation order) is kept so that seeded runs follow the same path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from qgf.downstream.gt.data import custom_collate_with_pe
from qgf.downstream.gt.model.config import ModelConfig
from qgf.downstream.gt.model.transformer import GraphTransformer


@dataclass
class Metrics:
    """Training/evaluation metrics."""

    mae: float = 0.0
    rmse: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0


LOSS_FUNCTIONS = {"l1": F.l1_loss, "mse": F.mse_loss, "cross_entropy": F.cross_entropy}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_target_value(data: Data) -> float:
    y = data.y
    return float(y.item()) if hasattr(y, "item") else float(y)


def build_warmup_plateau_schedulers(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, patience: int, factor: float = 0.5
) -> tuple[LinearLR | None, ReduceLROnPlateau]:
    """Linear warmup (start_factor 1e-6) then ReduceLROnPlateau."""
    warmup: LinearLR | None = None
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
    plateau = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
    return warmup, plateau


def _metrics(preds: torch.Tensor, targets: torch.Tensor, is_classification: bool) -> Metrics:
    if is_classification:
        pred_classes = preds.argmax(dim=-1).cpu().numpy()  # [total_samples]
        target_np = targets.squeeze().long().cpu().numpy()  # [total_samples]
        return Metrics(
            accuracy=float(accuracy_score(target_np, pred_classes)),
            f1_macro=float(f1_score(target_np, pred_classes, average="macro")),
        )
    return Metrics(
        mae=F.l1_loss(preds, targets).item(),
        rmse=torch.sqrt(F.mse_loss(preds, targets)).item(),
    )


def _loss(y_pred: torch.Tensor, y: torch.Tensor, loss_fn: str) -> torch.Tensor:
    if loss_fn == "cross_entropy":
        return F.cross_entropy(y_pred, y.squeeze().long())  # scalar
    return LOSS_FUNCTIONS[loss_fn](y_pred, y)  # scalar


def train_epoch(
    model: GraphTransformer,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    loss_fn: str = "l1",
) -> Metrics:
    model.train()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Training", leave=False, disable=None):
        batch = batch.to(device)
        optimizer.zero_grad()
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                y_pred = model(batch)  # [B] or [B, num_classes]
                loss = _loss(y_pred, batch.y, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(batch)  # [B] or [B, num_classes]
            loss = _loss(y_pred, batch.y, loss_fn)
            loss.backward()
            optimizer.step()
        all_preds.append(y_pred.detach().clone())
        all_targets.append(batch.y.detach().clone())
    return _metrics(torch.cat(all_preds), torch.cat(all_targets), loss_fn == "cross_entropy")


@torch.no_grad()
def evaluate(
    model: GraphTransformer,
    loader: TorchDataLoader,
    device: torch.device,
    is_classification: bool = False,
) -> Metrics:
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Evaluating", leave=False, disable=None):
        batch = batch.to(device)
        all_preds.append(model(batch).clone())
        all_targets.append(batch.y.clone())
    return _metrics(torch.cat(all_preds), torch.cat(all_targets), is_classification)


def train_simple(
    train_dataset: list[Data],
    val_dataset: list[Data],
    test_dataset: list[Data],
    config: ModelConfig,
    max_epochs: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 100,
    loss_fn: str = "l1",
    warmup_epochs: int = 0,
) -> tuple[Metrics, GraphTransformer, dict]:
    """Train with early stopping on the validation split; return test metrics at best val.

    Early stopping and ReduceLROnPlateau monitor val MAE (regression) or val macro-F1
    (classification). Returns ``(test_metrics, best_model, info)`` with
    ``info = {"best_epoch", "best_val", "epochs_run"}``.
    """
    device = get_device()
    print(f"Training on: {device}")

    def loader(ds: list[Data], shuffle: bool) -> TorchDataLoader:
        return TorchDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_with_pe,
        )

    train_loader = loader(train_dataset, True)
    val_loader = loader(val_dataset, False)
    test_loader = loader(test_dataset, False)

    # The original drew one batch for PE diagnostics here; this consumes the global
    # RNG (shuffle permutation), so it is kept to stay on the same seeded path.
    next(iter(train_loader))

    model = GraphTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler, plateau_scheduler = build_warmup_plateau_schedulers(
        optimizer, warmup_epochs=warmup_epochs, patience=50
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    is_classification = loss_fn == "cross_entropy"
    if is_classification:
        es_metric, es_maximize = "f1_macro", True
        best_val_metrics = Metrics(accuracy=0.0, f1_macro=0.0)
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=50)
    else:
        es_metric = "rmse" if loss_fn == "mse" else "mae"
        es_maximize = False
        best_val_metrics = Metrics(mae=float("inf"), rmse=float("inf"))

    def better(current: float, best: float) -> bool:
        return current > best if es_maximize else current < best

    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler, loss_fn)
        val_metrics = evaluate(model, val_loader, device, is_classification)
        val_value = getattr(val_metrics, es_metric)

        if warmup_scheduler is not None and epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_value)

        if epoch % 10 == 0 or epoch == 1:
            tr = getattr(train_metrics, es_metric)
            print(f"Epoch {epoch}/{max_epochs} | train {es_metric} {tr:.4f} | val {val_value:.4f}")

        if better(val_value, getattr(best_val_metrics, es_metric)):
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, is_classification)
    best_val = getattr(best_val_metrics, es_metric)
    print(f"Best val {es_metric}: {best_val:.4f} (epoch {best_epoch}) | test: {test_metrics}")
    return (
        test_metrics,
        model,
        {"best_epoch": best_epoch, "best_val": best_val, "epochs_run": epoch},
    )
