# ESA with gated fusion on MUTAG (paper Table 4)

Code: `src/qgf/esa/` (port of the internal fork of Edge Set Attention, Buterez et al.).
Needs a GPU (bf16) and either the `esa` extra on Linux (`uv sync --extra esa`: bitsandbytes,
for the 8-bit AdamW of the original runs) or, for bit-identical trials, the pinned
environment `paper_env/` (below).

- Model (`small`): edge-set attention, hidden 256, 8 heads, layers `S M M M M M S P`
  (masked attention between edges sharing a node), PMA with 32 seeds, gated MLPs (hidden
  512), Admin residuals with post-LayerNorm, graph embedding 128, one logit; no dropout.
- Inputs: one-hot atom (7) + cycle-basis ring membership (6) + RWSE (20 random-walk steps,
  BatchNorm + Linear to 24) on the nodes, one-hot bond (4) on the edges ("RWSE\*"), plus the
  quantum one-body / two-body blocks of the downstream benchmark (z-scored with training
  statistics).
- Fusion: `concat` feeds `[x_i, x_j, e_ij]` to one MLP; `gated` encodes the structural and
  the quantum parts separately and mixes them per dimension, `g * struct + (1 - g) * quantum`,
  gate bias initialised to 2 (`g` ~ 0.88).
- Protocol: MUTAG folds of the downstream benchmark (fold k test, k+1 validation), seeds
  1-5; 8-bit AdamW lr 5e-4, weight decay 1e-3, batch 32, <= 300 epochs, bf16, gradient-norm
  clipping 0.5, ReduceLROnPlateau (0.5, patience 37), early stopping (patience 75) and best
  checkpoint on the validation BCE; test metrics of the best checkpoint.
- Reported: binary F1 of the positive class (mean ± std ddof 0 over the 50 trials) and the
  mean learned gate (last forward pass, i.e. the test set).

## Commands

### Exact reproduction (pinned environment of the paper runs)

`paper_env/` is a uv project holding the package versions of the environment of the
original runs (Python 3.11, torch 2.4.0 + CUDA 12.1, xformers 0.0.27.post2, PyG 2.6.0,
PyTorch Lightning 2.5.3, torchmetrics 1.8.1, bitsandbytes 0.49.2, numpy 2.4.4,
scikit-learn 1.7.1); `qgf` is imported from the repository sources (`PYTHONPATH=src`).
With `--attention xformers` every trial is bit-identical to the stored trial of the
original runs on an A100 (see "Exactness" below).

```bash
uv sync --frozen --project reproduce/paper_env      # once (Linux x86_64, CUDA GPU)
# one trial (~1-2 min on an A100)
PYTHONPATH=src uv run --frozen --project reproduce/paper_env python \
    experiments/esa/run.py --attention xformers --configs af_fm --fusions gated \
    --folds 0 --seeds 1 --out results/esa_exact
# Table 4 (15 cells x 10 folds x 5 seeds = 750 trials), 16 GPUs x 4 trials per GPU;
# re-submitting resumes
ESA_ENV=paper NTASKS=16 PARALLEL=4 LOGS=results/esa_exact/logs \
    sbatch --array=0-15 experiments/esa/submit.sh --configs paper --out results/esa_exact
uv run python experiments/esa/analyze.py --results results/esa_exact \
    --expected experiments/esa/expected --tables experiments/esa/expected/tables/rerun \
    --summary-out experiments/esa/expected/rerun_mutag_summary.csv \
    --trials-out experiments/esa/expected/rerun_mutag_trials.csv
```

What the exactness needs, besides identical data, folds, seeds and initial weights:

- **Attention kernel**: xformers' memory-efficient attention (as the runs), not PyTorch SDPA.
- **SwiGLU**: the runs used flash-attn's fused SwiGLU (forward and backward in float32,
  one rounding to bf16); the port compiles the same kernel source with
  `torch.cuda.jiterator`, so flash-attn itself is not needed.
- **Training loop**: PyTorch Lightning exactly as the original `Estimator` (sanity
  validation pass, validation and test loaders iterated every epoch, which draws from the
  global RNG and shifts the shuffles; epoch-aggregated validation loss; torchmetrics on the
  bf16 logits), see `src/qgf/esa/trainer.py`.
- **One CPU thread** (`torch.set_num_threads(1)`) from the start of a trial, as the
  original: the z-scores and RWSE are reductions whose last bits depend on the thread count.

Running 4 trials concurrently on one GPU (`PARALLEL=4`) leaves the trials bit-identical.
Each trial directory keeps the best checkpoint (`best.ckpt`, ~70 MB; ~51 GB for the grid)
next to `results.json`; only `results.json` is needed by `analyze.py`.

### Project environment

```bash
uv sync --extra esa
uv run --extra esa python experiments/esa/run.py --configs af_fm --fusions gated --folds 0 --seeds 1
sbatch --array=0-7 experiments/esa/submit.sh --configs paper
uv run python experiments/esa/analyze.py --results results/esa --expected experiments/esa/expected
```

This runs with the repository's torch and PyTorch SDPA attention (`--attention sdpa`, the
default): the model and protocol are the same, but kernels and rounding differ, so single
trials are not bit-identical to the stored ones (compare cell statistics).

## Expected numbers

`expected/mutag_summary.csv` / `expected/mutag_trials.csv`: the original 750 trials
(`scripts/internal/esa_reference.py`). They reproduce every cell of Table 4 and every gate
value of its caption:

| method | concat | gated (gate) |
|---|---|---|
| RWSE\* | 0.905 ± 0.045 | — |
| RWSE\*+AFM<sub>stat</sub> | 0.899 ± 0.044 | 0.915 ± 0.042 (0.887) |
| RWSE\*+FM<sub>stat</sub> | 0.888 ± 0.076 | 0.918 ± 0.046 (0.855) |
| RWSE\*+AFM | 0.893 ± 0.049 | 0.915 ± 0.048 (0.862) |
| RWSE\*+FM | 0.913 ± 0.042 | 0.911 ± 0.043 (0.816) |
| RWSE\*+AFM+FM | 0.903 ± 0.050 | **0.922 ± 0.042** (0.830) |
| RWSE\*+HW-AFM | 0.907 ± 0.047 | 0.920 ± 0.044 (0.834) |
| RWSE\*+HW-AFM<sub>stat</sub> | 0.899 ± 0.046 | 0.909 ± 0.046 (0.861) |

## Exactness

The full grid re-run in the pinned environment (750 trials, 16 A100 GPUs x 4 trials per
GPU, about 1 h) is bit-identical to the original runs: **750/750 trials** have the same
test F1, accuracy, ROC-AUC, MCC, mean gate and parameter count as the stored trials (full
float32 values; `analyze.py --expected` checks them at the 6 decimals of the reference
CSV). Hence every cell, standard deviation and gate value of Table 4 is reproduced exactly:

- `expected/rerun_mutag_trials.csv`: the 750 re-run trials;
- `expected/rerun_mutag_summary.csv`: per-cell statistics (equal to `mutag_summary.csv`);
- `expected/tables/rerun/table4_esa.tex`: Table 4 from the re-run, identical to
  `expected/tables/paper/table4_esa.tex` except the provenance words of the caption.

## Notes on the original runs

- **Metric.** The stored and published value is the binary F1 of the positive class
  (torchmetrics `binary_f1_score`), not the macro-F1 stated in the Table 4 caption.
- **Standard deviation.** Table 4 uses ddof 0 over the 50 trials (column `std_ddof0`).
- **Hardware rows.** Configs `hw_afm_3t` / `hw_afm_3t_stat` (`afm_qpu*` in the original
  runs): the released `mutag_hw_afm` at 305, 512 and 4000 ns (without its 715 ns run; the
  values at the three other times are identical to those of the runs). The original loader
  read them in the ZZ convention (4 C) and clipped every one-body / two-body value to
  [-1, 1]; this clip is active for the hardware set (one-body values up to 1.3) and is
  reproduced for these configs (`src/qgf/downstream/gt/data.py`). Configs `hw_afm*` use
  all 4 times, without the clip.
- **Attention backend.** The runs used xformers' memory-efficient attention with an
  additive mask (0 between adjacent edges, -99999 elsewhere): `--attention xformers`. The
  default `--attention sdpa` applies the same mask with PyTorch's
  `scaled_dot_product_attention` (xformers is not a dependency of the project environment:
  it pins old torch versions). The fork's own PyTorch path inverted the mask (edges
  attending to non-adjacent edges); it was not used for the paper runs and is not
  reproduced.
- **Protocol.** Early stopping and checkpoint selection use the validation fold only (the
  test fold was passed to the trainer as a second validation loader, for logging).
