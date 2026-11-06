"""One- / two-body and model-size ablation on ZINC_tiny (Tables 8-10; port of the internal
``train_quantum_ablation.py``).

Data: ``zinc_tiny_{afm,fm}`` (886 / 211 / 219 graphs), the dataset of the original runs; its
features come from an earlier generation protocol (see the dataset's ``meta["time_note"]``).
The original runs fed the source arrays directly: one-body = occupations ``<n_i>(t)``,
two-body = connected correlations ``C_ij(t)`` including the diagonal ``n(1 - n)``; both are
rebuilt from the released ZZ-convention tensors and z-scored per time step with training
statistics (std clamped at 1e-6; two-body statistics over all ``N x N`` entries).

Configs: ``no_pe``, ``onebody_only``, ``twobody_only``, ``onebody_twobody``; ``zinc_mode``
``zinc`` adds the bond-type embedding to the edges ("all features"), ``no_zinc`` does not.
Model: :class:`qgf.downstream.gnns.model.GNNWithPE` (GCN or GAT) with 95 atom types, at one
of the sizes of Table 7, trained as the ZINC benchmark (AdamW, warmup, early stopping on
validation MAE, up to 1000 epochs). The paper's cells pool runs made with several settings
(:data:`SETTINGS`), see ``experiments/ablations/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from qgf.data import DEFAULT_ROOT, load_splits
from qgf.downstream.gnns.benchmark import HPARAMS, fit
from qgf.downstream.gnns.model import GNNWithPE

NUM_ATOM_TYPES = 95
NUM_BOND_TYPES = 4
DATASETS = {"ferro": "fm", "antiferro": "afm"}  # original names -> released pulse
CONFIGS = ["no_pe", "onebody_only", "twobody_only", "onebody_twobody"]
ZINC_MODES = ["zinc", "no_zinc"]
SIZES: dict[str, dict[str, list[int]]] = {
    "tiny": {"gnn_layers": [16, 16], "mlp_layers": [16, 16]},
    "small": {"gnn_layers": [32, 32], "mlp_layers": [32, 32]},
    "medium": {"gnn_layers": [32, 32, 32], "mlp_layers": [32, 32, 32]},
    "large": {"gnn_layers": [64] * 4, "mlp_layers": [64] * 4},
    "huge": {"gnn_layers": [128] * 4, "mlp_layers": [128] * 4},
    "very_huge": {"gnn_layers": [256] * 5, "mlp_layers": [256] * 3},
    "massive": {"gnn_layers": [512] * 4, "mlp_layers": [512] * 3},
    "giant": {"gnn_layers": [512] * 5, "mlp_layers": [512] * 3},
}
# Training settings of the original sweeps (CLI defaults of the original script = "v1").
SETTINGS: dict[str, dict[str, Any]] = {
    "v1": {"pooling": "add", "dropout": 0.0, "layer_norm": False, "num_heads": 1,
           "clamp": True, "shift": False},
    "v2_gat": {"pooling": "mean", "dropout": 0.1, "layer_norm": True, "num_heads": 1,
               "clamp": True, "shift": False},
    "v2_gat_16h": {"pooling": "mean", "dropout": 0.1, "layer_norm": True, "num_heads": 16,
                   "clamp": True, "shift": False},
    "v2_gcn_shift": {"pooling": "mean", "dropout": 0.1, "layer_norm": True, "num_heads": 1,
                     "clamp": False, "shift": True},
    "v2_gcn_clamp": {"pooling": "mean", "dropout": 0.1, "layer_norm": True, "num_heads": 1,
                     "clamp": True, "shift": False},
}  # fmt: skip


def source_arrays(g: Data, zz_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(occupation [T, N], C [T, N, N] with diagonal n(1 - n))`` of a released graph."""
    corr = g.twobody_features / zz_scale  # [T, N, N], zero diagonal
    idx = torch.arange(corr.shape[1])
    corr = corr.clone()
    corr[:, idx, idx] = g.onebody_features / zz_scale  # C_ii
    return g.occupation.float(), corr.float()


def load_zinc_tiny(dataset: str, root: Path = DEFAULT_ROOT) -> dict[str, list[Data]]:
    """Splits of ``zinc_tiny_<pulse>`` with ``ob [T, N]`` / ``tb [T, N, N]`` source arrays."""
    splits, meta = load_splits(f"zinc_tiny_{DATASETS[dataset]}", root=root)
    out = {}
    for k, graphs in splits.items():
        out[k] = []
        for g in graphs:
            ob, tb = source_arrays(g, float(meta["zz_scale_applied"]))
            out[k].append(
                Data(x=g.x.long(), edge_index=g.edge_index, edge_attr=g.edge_attr.long(),
                     y=g.y.float().view(-1), ob=ob, tb=tb)
            )  # fmt: skip
    return out


def normalisation(train: list[Data]) -> tuple[torch.Tensor, ...]:
    """Per-time-step mean / std (clamped at 1e-6) of one- and two-body arrays, train only."""
    ob = torch.cat([g.ob.T for g in train])  # [sum N, T]
    tb = torch.cat([g.tb.permute(1, 2, 0).reshape(-1, g.tb.shape[0]) for g in train])
    return ob.mean(0), ob.std(0).clamp(min=1e-6), tb.mean(0), tb.std(0).clamp(min=1e-6)


def transform(splits: dict[str, list[Data]], config: str) -> tuple[dict[str, list[Data]], int, int]:
    """Node-major z-scored ``onebody_features`` / ``twobody_features`` as the config needs."""
    ob_mean, ob_std, tb_mean, tb_std = normalisation(splits["train"])
    use_ob = config in ("onebody_only", "onebody_twobody")
    use_tb = config in ("twobody_only", "onebody_twobody")
    t = splits["train"][0].ob.shape[0]
    out = {}
    for k, graphs in splits.items():
        out[k] = []
        for g in graphs:
            d = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y)
            if use_ob:
                d.onebody_features = (g.ob.T - ob_mean) / ob_std  # [N, T]
            if use_tb:
                d.twobody_features = (g.tb.permute(1, 2, 0) - tb_mean) / tb_std  # [N, N, T]
            out[k].append(d)
    return out, (t if use_ob else 0), (t if use_tb else 0)


def run_trial(
    model_type: str,
    dataset: str,
    config: str,
    size: str,
    zinc_mode: str,
    setting: str,
    seed: int,
    output_dir: Path | None = None,
    root: Path = DEFAULT_ROOT,
    max_epochs: int | None = None,
    splits: dict[str, list[Data]] | None = None,
) -> dict[str, Any]:
    """One ablation trial; ``splits`` passes an in-memory dataset (tests)."""
    if model_type not in ("GCN", "GAT") or setting not in SETTINGS or config not in CONFIGS:
        raise ValueError(f"unknown model / setting / config: {model_type}, {setting}, {config}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw = splits if splits is not None else load_zinc_tiny(dataset, root)
    data, abs_dim, rel_dim = transform(raw, config)
    torch.manual_seed(seed)  # re-seed before model initialisation, as the original
    np.random.seed(seed)
    s = SETTINGS[setting]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNWithPE(
        model_type,
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        abs_pe_dim=abs_dim,
        rel_pe_dim=rel_dim,
        output_dim=1,
        pooling=s["pooling"],
        dropout=s["dropout"],
        num_heads=s["num_heads"],
        layer_norm=s["layer_norm"],
        edge_weight_clamp=(1e-8, 100.0) if s["clamp"] else None,
        edge_weight_shift=s["shift"] and model_type == "GCN",
        bond_features=zinc_mode == "zinc",
        **SIZES[size],
    ).to(device)
    hp = dict(HPARAMS["zinc"])
    if max_epochs is not None:
        hp["max_epochs"] = max_epochs
    metrics = fit(model, data, False, **hp)
    results = {
        "model_type": model_type,
        "dataset": dataset,
        "config": config,
        "size": size,
        "zinc_mode": zinc_mode,
        "setting": setting,
        "seed": seed,
        "abs_pe_dim": abs_dim,
        "rel_pe_dim": rel_dim,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **metrics,
        "training_config": hp,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
