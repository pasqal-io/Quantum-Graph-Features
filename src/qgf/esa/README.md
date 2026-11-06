# Edge Set Attention with gated fusion (Table 4)

Port of the ESA model and MUTAG training used for Table 4. Experiment, commands and expected
numbers: [`experiments/esa/README.md`](../../../experiments/esa/README.md).

| module | content |
|---|---|
| `layers.py` | MAB / SAB / PMA attention with the original additive edge-adjacency mask (xformers' memory-efficient kernel as the paper runs, or PyTorch SDPA), gated MLPs with flash-attn's fused SwiGLU kernel (compiled with `torch.cuda.jiterator`), Admin residuals, the ESA encoder-decoder |
| `model.py` | `ESAClassifier`: RWSE encoder, edge tokens by concatenation or gated fusion, ESA, logit |
| `trainer.py` | the PyTorch Lightning training loop of the original runs (8-bit AdamW, bf16, gradient clipping, ReduceLROnPlateau, early stopping and best checkpoint on the validation loss), torchmetrics binary metrics |
| `benchmark.py` | one trial: features (atoms, rings, RWSE, quantum blocks), model, training, `results.json` |

In the pinned environment `reproduce/paper_env` with `attention="xformers"`, trials are
bit-identical to the original runs (all 750 trials of Table 4).
