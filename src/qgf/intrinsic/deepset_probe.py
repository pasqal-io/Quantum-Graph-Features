"""DeepSet probes without message passing (ported from the original internal implementation).

A validation split is held out from the training graphs of each fold; early
stopping and checkpoint selection use it, the test fold is only used for the
final score.

``pe`` may be a list of per-graph arrays or a callable ``train_idx -> list``
that builds the features of all graphs for one fold (used to fit the feature
normalisation on the training graphs only).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

VAL_FRACTION = 0.1


class SimpleDeepSet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(self.phi(x))


class GraphDeepSet(nn.Module):
    """DeepSet for graph-level prediction: phi per node -> sum-pool -> rho."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        num_graphs = int(batch.max()) + 1
        pooled = torch.zeros(num_graphs, h.size(1), device=h.device)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
        return self.rho(pooled).squeeze(-1)


class _EntityDataset(Dataset):
    """Per-entity samples flattened across graphs; indexed by a list of indices."""

    def __init__(self, pe_list: list[np.ndarray], targets: list[np.ndarray]) -> None:
        self.features = torch.from_numpy(np.concatenate(pe_list, axis=0)).float()
        self.targets = torch.from_numpy(np.concatenate(targets, axis=0)).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def _entity_loader(ds: _EntityDataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Same batches and RNG draws as ``DataLoader(ds, batch_size, shuffle)``, without
    per-sample collation (one tensor gather per batch)."""
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(
        ds, sampler=BatchSampler(sampler, batch_size, drop_last=False), batch_size=None
    )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inner_split(train_idx: np.ndarray, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out VAL_FRACTION of the training graphs for early stopping."""
    rng = np.random.RandomState(seed + fold)
    perm = rng.permutation(train_idx)
    n_val = max(1, round(VAL_FRACTION * len(perm)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def _fit_predict(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    patience,
    step,
    device,
):
    """Train with early stopping on ``val_loader``; return (preds, true) on ``test_loader``."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for _ in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds, yb = step(model, batch)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                preds, yb = step(model, batch)
                val_losses.append(criterion(preds, yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds, yb = step(model, batch)
            all_preds.append(preds.cpu())
            all_true.append(yb.cpu())
    return torch.cat(all_preds), torch.cat(all_true)


def _fold_pe(pe, train_idx):
    return pe(train_idx) if callable(pe) else pe


def evaluate_deepset_cv(
    pe: list[np.ndarray] | Callable,
    targets: list[np.ndarray],
    classification: bool = False,
    seed: int = 42,
    num_epochs: int = 50,
) -> tuple[float, float, list[float]]:
    """Node/edge probe: 5-fold CV over graphs, samples = all entities of the graphs.

    Returns (mean, population std, per-fold scores); accuracy or MAE.
    """
    _seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim, lr, patience, batch_size = 64, 1e-3, 10, 512

    graph_indices = np.arange(len(targets))
    if classification:
        graph_labels = [int(t.mean() > 0.5) for t in targets]
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = kf.split(graph_indices, graph_labels)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = kf.split(graph_indices)

    criterion = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()

    def step(model, batch):
        xb, yb = batch[0].to(device), batch[1].to(device)
        return model(xb).squeeze(-1), yb

    scores = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        pe_list = _fold_pe(pe, train_idx)
        fit_idx, val_idx = _inner_split(train_idx, seed, fold)

        def loader(idx, shuffle, pe_list=pe_list):
            ds = _EntityDataset([pe_list[i] for i in idx], [targets[i] for i in idx])
            return _entity_loader(ds, batch_size, shuffle)

        train_loader = loader(fit_idx, True)
        test_loader = loader(test_idx, False)
        val_loader = loader(val_idx, False)

        model = SimpleDeepSet(pe_list[0].shape[1], hidden_dim, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        preds, true = _fit_predict(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            patience,
            step,
            device,
        )
        if classification:
            preds_bin = (torch.sigmoid(preds) > 0.5).int().numpy()
            score = float(accuracy_score(true.numpy(), preds_bin))
        else:
            score = float(torch.mean(torch.abs(preds - true)).item())
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores)), scores


def _collate_graphs(batch):
    """Collate variable-size graphs into a batched representation."""
    xs, ys = zip(*batch)
    lengths = [x.size(0) for x in xs]
    x_cat = torch.cat(xs, dim=0)
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)])
    y_cat = torch.stack(ys)
    return x_cat, batch_idx, y_cat


class _GraphDataset(Dataset):
    def __init__(self, pe_list: list[np.ndarray], targets: np.ndarray) -> None:
        self.pe_list = pe_list
        self.targets = targets

    def __len__(self) -> int:
        return len(self.pe_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.pe_list[idx]).float()
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def evaluate_deepset_graph_cv(
    pe: list[np.ndarray] | Callable,
    targets: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    hidden_dim: int = 64,
    num_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[float, float, list[float]]:
    """Graph probe (sum-pooled DeepSet), KFold over graphs, MSE loss, MAE score.

    The original never passed ``classification`` here: every graph task is a regression.
    """
    _seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = np.arange(len(targets))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    criterion = nn.MSELoss()

    def step(model, batch):
        x_cat, batch_idx, y = (t.to(device) for t in batch)
        return model(x_cat, batch_idx), y

    scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        pe_list = _fold_pe(pe, train_idx)
        fit_idx, val_idx = _inner_split(train_idx, seed, fold)

        def loader(idx, shuffle, pe_list=pe_list):
            ds = _GraphDataset([pe_list[i] for i in idx], targets[idx])
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_graphs
            )

        train_loader = loader(fit_idx, True)
        test_loader = loader(test_idx, False)
        val_loader = loader(val_idx, False)

        model = GraphDeepSet(pe_list[0].shape[1], hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        preds, true = _fit_predict(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            num_epochs,
            patience,
            step,
            device,
        )
        scores.append(float(torch.mean(torch.abs(preds - true)).item()))

    return float(np.mean(scores)), float(np.std(scores)), scores
