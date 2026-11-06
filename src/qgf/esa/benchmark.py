"""One trial of the ESA MUTAG benchmark (Table 4; port of ``train_mutag_with_pe.py``).

Data, folds and quantum encodings are the downstream benchmark's
(:mod:`qgf.downstream.gt`): 10-fold CV stratified by (label, size quartile), fold ``k``
test, fold ``k+1`` validation, seeds 1-5, z-scored AFM / FM blocks. On top of them:

- nodes: one-hot atom (7) + cycle-basis ring membership (6) + quantum one-body block +
  RWSE (random-walk landing probabilities, 20 steps; encoded inside the model);
- edges: one-hot bond (4) + quantum two-body block of the edge.

Training: the original PyTorch Lightning loop (:mod:`qgf.esa.trainer`): 8-bit AdamW
(bitsandbytes; plain AdamW on CPU), lr 5e-4, weight decay 1e-3, batch 32, up to 300
epochs, bf16 on GPU, gradient-norm clipping 0.5, BCE-with-logits; ReduceLROnPlateau
(factor 0.5, patience 37), early stopping (patience 75) and best checkpoint on the
validation loss; test metrics of the best checkpoint. Reported metric: binary F1 of the
positive class (as stored by the original runs), plus accuracy, ROC-AUC, MCC and the mean
fusion gate of the last forward.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from qgf.data import DEFAULT_ROOT
from qgf.downstream.gt.benchmark import build_splits, transform_splits
from qgf.downstream.gt.pe_transforms import compute_ring_cycle_basis_pe
from qgf.esa.layers import set_attention_backend
from qgf.esa.model import RWSE_STEPS, ESAClassifier
from qgf.esa.trainer import fit

# hw_afm_3t*: the hardware set at 305, 512 and 4000 ns, as the Table 4 runs (hw_afm*: all 4)
CONFIGS = ["no_pe", "af", "af_stat", "fm", "fm_stat", "af_fm", "hw_afm_3t", "hw_afm_3t_stat",
           "hw_afm", "hw_afm_stat"]  # fmt: skip
FUSIONS = ["concat", "gated"]
SEEDS = [1, 2, 3, 4, 5]
NUM_ATOM_TYPES, NUM_BOND_TYPES = 7, 4
HPARAMS: dict[str, Any] = {
    "max_epochs": 300,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "patience": 75,
    "gradient_clip_val": 0.5,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rw_landing_probs(edge_index: torch.Tensor, num_nodes: int, steps: int = RWSE_STEPS):
    """``[N, steps]`` diagonals of ``(D^-1 A)^k``, k = 1..steps."""
    deg = torch.zeros(num_nodes).scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]
    out, Pk = [], P.clone()
    for _ in range(steps):
        out.append(torch.diagonal(Pk))
        Pk = Pk @ P
    return torch.stack(out, dim=1)  # [N, steps]


def esa_graph(d: Data) -> Data:
    """Node / edge features of one transformed graph (see module docstring)."""
    parts = [F.one_hot(d.x.long().view(-1), NUM_ATOM_TYPES).float()]
    parts.append(compute_ring_cycle_basis_pe(d)[0])  # [N, 6]
    ob = getattr(d, "onebody_features", None)
    if ob is not None:
        parts.append(ob.float())
    edge = [F.one_hot(d.edge_attr.long().view(-1), NUM_BOND_TYPES).float()]
    tb = getattr(d, "twobody_features", None)
    if tb is not None:
        edge.append(tb[d.edge_index[0], d.edge_index[1]].float())  # [E, Q2]
    return Data(
        x=torch.cat(parts, dim=1),
        edge_index=d.edge_index,
        edge_attr=torch.cat(edge, dim=1),
        y=d.y.view(-1).float(),
        pestat_RWSE=rw_landing_probs(d.edge_index, d.x.shape[0]),
        initial_id=d.initial_id,
    )


def prepare(splits: dict[str, list[Data]]) -> tuple[dict[str, list[Data]], int]:
    max_edges = max(d.edge_index.shape[1] for s in splits.values() for d in s)
    return {k: [esa_graph(d) for d in v] for k, v in splits.items()}, max_edges


def run_trial(
    config: str,
    fusion: str,
    seed: int,
    fold: int,
    output_dir: Path | None = None,
    root: Path = DEFAULT_ROOT,
    max_epochs: int | None = None,
    graphs: list[Data] | None = None,
    model_kwargs: dict | None = None,
    attention: str = "sdpa",
) -> dict[str, Any]:
    """Run one (config, fusion, fold, seed) MUTAG trial; write ``results.json``.

    ``max_epochs`` / ``model_kwargs`` override the paper values (smoke tests);
    ``graphs`` passes an in-memory MUTAG-like dataset; ``attention="xformers"`` uses the
    kernel of the paper runs (bit-identical trials in the pinned environment
    ``reproduce/paper_env``).
    """
    if config not in CONFIGS or fusion not in FUSIONS:
        raise ValueError(f"unknown config / fusion {config!r} / {fusion!r}")
    set_attention_backend(attention)
    # one CPU thread from the start, as the original: the feature transforms (z-scores,
    # RWSE) are reductions whose last bits depend on the thread count
    torch.set_num_threads(1)
    seed_everything(seed)
    raw = build_splits("mutag", config, fold, root, graphs)
    splits, abs_dim, rel_dim = transform_splits(config, raw)
    splits, max_edges = prepare(splits)
    hp = dict(HPARAMS)
    if max_epochs is not None:
        hp["max_epochs"] = max_epochs
    loaders = {
        k: DataLoader(v, batch_size=hp["batch_size"], shuffle=(k == "train"))
        for k, v in splits.items()
    }
    next(iter(loaders["train"]))  # the original drew one batch to check dimensions (RNG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = splits["train"][0].x.shape[1]
    model = ESAClassifier(
        num_features=num_features,
        edge_dim=splits["train"][0].edge_attr.shape[1],
        set_max_items=max_edges,
        fusion=fusion,
        quantum_node_dim=abs_dim,
        quantum_edge_dim=rel_dim,
        **(model_kwargs or {}),
    ).to(device)
    metrics = fit(model, loaders, device, **hp, output_dir=output_dir)
    results: dict[str, Any] = {
        "dataset": "MUTAG",
        "model_type": "ESA",
        "config": config,
        "fusion": model.fusion,
        "fold": fold,
        "seed": seed,
        "abs_pe_dim": abs_dim,
        "rel_pe_dim": rel_dim,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gate_mean": model.last_gate_mean,
        **metrics,
        "training_config": hp,
        "attention": attention,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
