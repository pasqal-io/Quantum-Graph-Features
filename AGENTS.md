---
protocol_version: "0.1"
title: "Neutral-Atom Quantum Features as Complementary Structural Encodings for Graph Learning"
authors: "Mathieu Garrigues, Igor Sokolov, Matthias Hecker, Lorenzo Moro, Shaheen Acheche (Pasqal, France)"
domain: "graph machine learning; neutral-atom quantum computing"
version: "0.1.0"
---

# Agent notes (tacit knowledge only; see README.md for the basics)

Paper-item -> command map: `reproduce/README.md`. Explanations and reproduction workflow:
skill `.claude/skills/qgf-paper/`. Each `experiments/<item>/README.md` has exact commands
and expected values.

## Data conventions
- Correlations are stored in the **ZZ convention, `4 C`** with `C = <n_i n_j> - <n_i><n_j>`,
  for every dataset (`meta["zz_scale_applied"]` = factor applied at export: 1 for MUTAG
  emulation and MUTAG residuals, 4 for ZINC, ZINC1500 and QPU). Divide by it to get the
  original code's inputs; most pipelines z-score or min-max scale, so the factor is a no-op.
- Layout is **time-major**: `onebody_features [T, N]`, `twobody_features [T, N, N]` (zero
  diagonal). PyG batching is node-major: transpose explicitly, always (never decide by
  comparing shapes: N can equal T = 20).
- One-body = diagonal of C (`n(1-n)`), not the occupation.
- Times: `meta["times_ns"]`, emulation `linspace(100, 4000, 20, int)`. QPU sets have 4
  nominal times `[305, 512, 715, 4000]`; only 305 and 715 are emulation-equivalent (see
  `meta["time_note"]`). The reconstruction reference instant is time index 7 (1536 ns).
- AFM and FM files are index-aligned by `initial_id`. ZINC\*_subset = 11 368 of the
  11 494 embeddable ZINC graphs, splits 9475 / 946 / 947.
- Bit-identical reproduction (ESA, downstream GNNs, ablations, non-unit-disk) needs the pinned
  env `reproduce/paper_env` (`PYTHONPATH=src`), the CPU / GPU model and the thread count of the
  original runs: GNNs and ablations on AMD EPYC 7742 nodes without GPU (`-p internal`, no
  `--gres`; the AVX-512 EPYC 9654 nodes of `-p cpu` change the last bits); MUTAG GNNs and
  non-unit-disk 1 thread per process, ZINC GNNs 4 (1 for the Laplacian / geometry rows and 30
  resubmitted trials), ablations 4. Wrappers: `ESA_ENV=paper`, `GNN_ENV=paper`, `ABL_ENV=paper`.
  `GNNWithPE` embeds bonds per graph when two-body features are present (keep it: the gradient
  sum order decides the last bit).
- Downstream GNNs (`experiments/downstream_gnns`): their geometry baseline is
  `rrwp_lap_ring_geom*` (cycle-basis rings, 5 / 3 geometry channels), not the GT's
  `rrwp_lap_ring_geo*`; cells are mean ± std (ddof 1) over trials, but the paper's MUTAG
  hardware rows use the std of fold means (`std_fold`). The paper's GNN Laplacian row came
  from a run with a 1-dim Laplacian (`expected/*_paper_laplacian_v2.csv`).
- `ladder_{afm,fm}` (synthetic ladders): expectation values (no shots), float64, emu-mps
  bond 16; `occupation [T, N]` holds `<n_i>` (the benchmark uses it, not `n(1-n)`);
  `cv_fold [3]` are the reference folds (scikit-learn >= 1.8 reshuffles
  `StratifiedGroupKFold`, so never regenerate them to match the table); registers are the
  exact unit-disk layout of `qgf.synthetic.ladder.unit_disk_positions`, not Blade.
- `mutag_hw_afm` = QPU features; the one-body diagonal uses the single-atom readout
  correction of the jackknife estimator (`qgf.generate.correlations`).
- ESA (Table 4, `experiments/esa`): the metric is binary F1 (positive class), not macro F1,
  std ddof 0; GPU, `--extra esa` (bitsandbytes, Lightning; Linux). Bit-identical trials need
  the pinned env `reproduce/paper_env` (torch 2.4.0+cu121, xformers 0.0.27.post2,
  Lightning 2.5.3), `PYTHONPATH=src`, `--attention xformers` (or `ESA_ENV=paper` with
  `submit.sh`; 4 trials per GPU are safe); the trial sets one CPU thread before any data
  processing (thread count changes the last bits of the features). `hw_afm_3t*` =
  `mutag_hw_afm` at 305 / 512 / 4000 ns, ZZ scale kept and clipped to [-1, 1]: do not drop
  the clip (it is active and changes the HW rows).
- `zinc_tiny_{afm,fm}` (Tables 8-10): an earlier protocol (6 us, fixed (Omega, delta)),
  886 / 211 / 219 graphs; `occupation` kept; the paper cells pool several training settings
  and repeated seeds (`experiments/ablations/expected/trials.csv`), keep that pooling.
- Fig. 6 original runs selected the model on the test split (and their parameter labels
  are wrong); the port selects on validation, so it is not expected to equal the figure.
- Fig. 2 (`experiments/reconstruction/fig2.py`): the published FM panel used `delta =
  delta_inf`, not the paper's `2 delta_inf` (`--fm-delta paper`).
- `mutag_exact[_nn]_{afm,fm}` (non-unit-disk): exact emu-sv expectations, 173 graphs with
  N <= 24, times `linspace(100, 4000, 20)` unrounded; `_nn` = graph Hamiltonian `J_NN A`
  passed as interaction matrix; regenerated files equal the original features to 3e-8
  (1.2-1.7 A100 h per file). Benchmarks on them use `subset="exact173"` for every config;
  with a subset, `load_mutag` restricts every source (incl. the 188-graph emulation files).

## Where things are
- `src/qgf/data` loaders, manifest, Zenodo download; `src/qgf/generate` sequences,
  emulation, correlation estimators, QPU helpers; `src/qgf/intrinsic`, `reconstruction`,
  `downstream/gt`, `downstream/gnns` (other backbones, ablation), `esa`, `stats`; slot
  `synthetic` (README only).
- `experiments/<item>/` entry points + `expected/` reference tables; `results/` outputs
  (gitignored); `data/` (gitignored, from Zenodo).
- `scripts/internal/` needs internal infrastructure: never required to reproduce.
- `data/raw` layout: `emulation/{mutag,zinc}/`, `qpu/mutag_afm/` (only the four runs of
  `mutag_hw_afm`, `qgf.generate.raw.QPU_RUNS`), `embeddings/`, `layout_sensitivity/`;
  always resolve names with `qgf.generate.raw.raw_path`.
- `scripts/zenodo_upload.py` (+ `scripts/zenodo/` record README/MANIFEST generator): with
  `--deposition-id` it only adds missing/changed files, never the record metadata
  (unless `--set-metadata`); `--replace NAME...` deletes those draft files first (only
  files it uploads). Uploads `qgf_raw.tar` and `qgf_results.tar` (allow-list `RESULTS`);
  both are built only if absent: delete them to rebuild.

## Commands
```bash
uv sync [--extra stats] [--extra generate]
uv run pytest tests/<area> -q        # areas: data, generate, intrinsic, reconstruction, downstream, stats, synthetic
uv run ruff check
uv run python -c "from qgf.data import verify; print(verify())"   # [] = data ok
```
Heavy grids (intrinsic, downstream, ZINC emulation) want A100 GPUs; Slurm wrappers sit
next to each `run.py`. Stan (`stats` extra) needs libstdc++ from GCC >= 11.

## Not in this repo
Physics section (Figs. 1, 3, App. B-C): to be decided. Graph embedding code: use Blade (`qoolqit.embedding.Blade`). Re-running ZINC
feature generation costs ~800 A100 GPU-days; the QPU run cannot be repeated from here.
