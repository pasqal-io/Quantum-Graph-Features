"""Training of the ESA classifier with PyTorch Lightning, as the original runs (Table 4).

The original trained the ESA ``Estimator`` (a ``LightningModule``) with a Lightning
``Trainer``; this module reproduces that loop step for step, because its side effects
matter for bit-identical trials: the sanity-check validation pass and the validation and
test loaders iterated every epoch (``val_dataloaders=[val, test]``) each draw from the
global RNG when an iterator is created, which shifts the training shuffles; the monitored
``val_loss/dataloader_idx_0`` is Lightning's epoch aggregate of the per-batch losses;
metrics are torchmetrics' binary metrics on the bf16 logits of the epoch.

- optimiser: bitsandbytes 8-bit AdamW (plain AdamW on CPU), ReduceLROnPlateau on the
  validation loss (factor 0.5, patience ``patience // 2``);
- ``Trainer(max_epochs, precision="bf16", gradient_clip_val)``, ``ModelCheckpoint`` (best
  validation loss) and ``EarlyStopping`` (``patience``) on ``val_loss/dataloader_idx_0``;
- test: ``trainer.test(ckpt_path="best")``; F1 / accuracy / MCC at 0.5 and ROC-AUC of the
  positive class; the gate value of the last forward pass (the test set).
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

MONITOR = "val_loss/dataloader_idx_0"


def binary_metrics_pt(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(AUROC, MCC, accuracy, F1)``; torchmetrics applies a sigmoid to logits."""
    cls = torchmetrics.functional.classification
    y_true = y_true.to(torch.long)
    return (
        cls.binary_auroc(y_pred, y_true, thresholds=None),
        cls.binary_matthews_corrcoef(y_pred, y_true, threshold=0.5),
        cls.binary_accuracy(y_pred, y_true, threshold=0.5),
        cls.binary_f1_score(y_pred, y_true, threshold=0.5),
    )


class ESAModule(pl.LightningModule):
    """Lightning wrapper of :class:`qgf.esa.model.ESAClassifier` (binary task)."""

    def __init__(self, model: torch.nn.Module, lr: float, weight_decay: float, patience: int,
                 batch_size: int) -> None:  # fmt: skip
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)
        self.train_metrics, self.val_metrics, self.val_test_metrics = {}, {}, {}
        self.test_metrics: dict[int, tuple] = {}
        self.num_called_test = 1

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch)

    def configure_optimizers(self):
        if self.device.type == "cuda":
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.patience // 2
        )
        return {"optimizer": opt, "monitor": MONITOR, "lr_scheduler": sched}

    def _step(self, batch, step_type: str) -> torch.Tensor:
        predictions = self(batch)
        y = batch.y.view(predictions.shape)
        loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())
        predictions = predictions.detach().squeeze()
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        output = (predictions.cpu(), y.cpu())
        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        else:
            self.test_output[self.num_called_test].append(output)
        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self._step(batch, "train")
        if loss:
            self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss = self._step(batch, "validation")
            self.log("val_loss", loss)
            return loss
        loss = self._step(batch, "validation_test")
        self.log("val_test_loss", loss)
        return loss

    def test_step(self, batch, batch_idx: int):
        loss = self._step(batch, "test")
        self.log("test_loss", loss)
        return loss

    def _epoch_end_report(self, outputs, epoch_type: str):
        y_pred = torch.cat([o[0] for o in outputs], dim=0)
        y_true = torch.cat([o[1] for o in outputs], dim=0)
        metrics = binary_metrics_pt(y_true, y_pred)
        for name, value in zip(("AUROC", "MCC", "Accuracy", "F1"), metrics):
            self.log(f"{epoch_type} {name}", value, batch_size=self.batch_size)
        return metrics, y_pred, y_true

    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], _, _ = self._epoch_end_report(
            self.train_output[self.current_epoch], "Train"
        )
        del self.train_output[self.current_epoch]

    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], _, _ = self._epoch_end_report(
                self.val_output[self.current_epoch], "Validation"
            )
            del self.val_output[self.current_epoch]
        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], _, _ = self._epoch_end_report(
                self.val_test_output[self.current_epoch], "ValidationTest"
            )
            del self.val_test_output[self.current_epoch]

    def on_test_epoch_end(self):
        self.test_metrics[self.num_called_test], _, _ = self._epoch_end_report(
            self.test_output[self.num_called_test], "Test"
        )
        self.num_called_test += 1


def fit(
    model: torch.nn.Module,
    loaders: dict[str, Any],
    device: torch.device,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    gradient_clip_val: float,
    batch_size: int,
    output_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Train with the original Trainer configuration; test metrics of the best checkpoint."""
    module = ESAModule(model, learning_rate, weight_decay, patience, batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(output_dir) if output_dir is not None else Path(tmp)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            precision="bf16" if device.type == "cuda" else "32-true",
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="best",
                    monitor=MONITOR,
                    mode="min",
                    save_top_k=1,
                ),  # fmt: skip
                EarlyStopping(monitor=MONITOR, patience=patience, mode="min"),
            ],
            logger=False,
            default_root_dir=str(ckpt_dir),
            gradient_clip_val=gradient_clip_val,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, train_dataloaders=loaders["train"],
                    val_dataloaders=[loaders["val"], loaders["test"]])  # fmt: skip
        stopped = trainer.current_epoch
        best = trainer.checkpoint_callback.best_model_score
        trainer.test(model=module, dataloaders=loaders["test"], ckpt_path="best", verbose=False)
    auroc, mcc, acc, f1 = (float(v) for v in module.test_metrics[1])
    return {
        "best_val_loss": None if best is None else float(best),
        "epochs_run": stopped,
        "test_f1": f1,
        "test_accuracy": acc,
        "test_auc": auroc,
        "test_mcc": mcc,
    }
