# Neutral-atom quantum graph features: datasets

Data accompanying the NeurIPS 2026 paper *Neutral-Atom Quantum Features as Complementary
Structural Encodings for Graph Learning* (Mathieu Garrigues, Igor Sokolov, Matthias Hecker,
Lorenzo Moro, Shaheen Acheche; Pasqal, France). DOI: 10.5281/zenodo.23099924.
Code: the `qgf` repository (Python package `qgf`), which downloads, verifies and loads
these files.

## File structure

- `*.pt`: processed datasets, one file per dataset (and per split for ZINC), loaded by
  `qgf.data.load`. `manifest.json` lists their sha256 and number of graphs.
- `qgf_raw.tar`: uncompressed tar of `raw/` (the `.npz` inside are already compressed):
  - `raw/emulation/mutag/{afm,fm}.npz`, `raw/emulation/zinc/{afm,fm}_{train,val,test}.npz`:
    bitstring counts of the emulation runs;
  - `raw/qpu/mutag_afm/{t2_305ns,t2p5_512ns,t4_715ns,t15_4000ns}.npz`: bitstring counts of
    the four QPU runs of `mutag_hw_afm` (T2, T2.5, T4, T15; T4 is Fig. 5);
  - `raw/embeddings/zinc_{train,val,test}.npz`: ZINC register positions and graph structure;
  - $n_layout layout-sensitivity files under `raw/layout_sensitivity/`.
  Extract with `tar xf qgf_raw.tar -C data`.
- `qgf_results.tar`: the $n_results per-trial / per-cell result files (`.json`, `.csv`,
  `.parquet`) of the reference runs of every experiment of the `qgf` repository, under
  `results/` (layout and producing commands in `results/README.md` inside the archive).
  Extract with `tar xf qgf_results.tar`.
- `qgf_results_additions.tar`: per-trial `results.json` of the bit-identical re-runs of the
  downstream GNNs (Tables 2-3), ESA (Table 4), the ablations (Tables 8-10) and the exact
  non-unit-disk benchmark, under `results/{downstream_gnns,esa,ablations,non_unit_disk}/`.
  Extract with `tar xf qgf_results_additions.tar`.
- `MANIFEST.csv`: every file (including each member of `qgf_raw.tar` and
  `qgf_results.tar`, as `raw/...` and `results/...`) with
  size, sha256, md5, number of graphs, kind and a short description.
- `LICENSE.md`: licence.

| file | kind | graphs | MB |
|---|---|---|---|
$file_table

## Provenance

- **Emulation** (`kind = emulation`): the AFM and FM pulse sequences were simulated with
  the public `pulser`, `emu-sv` and `emu-mps` packages from the stored atom positions;
  the counts can be regenerated with `qgf.generate`.
- **QPU** (`kind = qpu`): measurements of the MUTAG AFM sequence on a Pasqal neutral-atom
  QPU (four runs, nominal durations 305, 512, 715 and 4000 ns; `meta["run"]` gives the
  pulse actually played).
- **Residual** (`kind = residual`): features with the part explained by classical graph
  descriptors removed (see the paper, reconstruction analysis).
- Features are connected correlations estimated from the counts with the corrected
  jackknife estimators of `qgf.generate`; for the QPU data the one-body (diagonal) term
  uses the single-atom readout correction.

## Formats

- `*.pt`: weights-only torch files (`torch.load(path, weights_only=True)`) holding a plain
  dict `{"format_version", "meta", "graphs"}`; `graphs` is a list of dicts of tensors:
  `x`, `edge_index`, `edge_attr`, `y`, `pos` [N, 2] (atom positions, um), `initial_id`,
  `onebody_features` [T, N], `twobody_features` [T, N, N] (zero diagonal), float32.
  `ladder_{afm,fm}.pt` (synthetic ladders) are float64 expectation values (no shots) and
  also hold `occupation` [T, N], `m`, `crossings`, `degeneracy`, `blind_group`,
  `blind_group_k3` and `cv_fold` (reference cross-validation folds).
  `mutag_exact[_nn]_{afm,fm}.pt` (173 MUTAG graphs with N <= 24) are exact state-vector
  expectation values (no shots) at `linspace(100, 4000, 20)` ns, with register couplings
  (`mutag_exact_*`) or the graph Hamiltonian `J_NN A` (`mutag_exact_nn_*`), and also hold
  `occupation` [T, N]. `zinc_tiny_{afm,fm}_{train,val,test}.pt` (ablations) hold the features of
  a 6 us constant pulse (see `meta["time_note"]`) and `occupation` [T, N].
  `meta`: `name`, `split`, `times_ns`, `zz_scale_applied`, `kind`, optional `time_note`.
- Raw counts (`raw/emulation/**`, `raw/qpu/**`, numpy only): `initial_id` [G], `num_nodes` [G], `times_ns` [T],
  `offsets` [G*T + 1], `bits` uint64 [K], `counts` uint32 [K], `meta` (JSON string).
  Record `(g, t)` (row `r = g*T + t`) owns `bits[offsets[r]:offsets[r+1]]` and the
  matching `counts`; bit `i` (LSB first) is atom `i` in the Rydberg state.
  `qgf.generate.raw.load_raw` reads them back as `{bitstring: count}` dicts.
- `raw/embeddings/zinc_<split>.npz`: register positions (`pos`) and graph structure.

## Conventions

- Correlations use the ZZ convention: `ZZ = 4 C`, with `C = <n_i n_j> - <n_i><n_j>`.
- Time-major layout: `[T, N]` and `[T, N, N]` (transpose to node-major for PyG batching).
- `meta["times_ns"]`: measurement times in ns (emulation: 20 times from 100 to 4000 ns).
- AFM and FM files are index-aligned by `initial_id`.

## Checksums

`manifest.json` (sha256 of the processed files) and `MANIFEST.csv` (sha256 and md5 of
every file). Zenodo also publishes the md5 of each uploaded file.

## Loading

```python
from qgf.data import load, load_splits, verify

graphs, meta = load("mutag_afm")  # list[torch_geometric.data.Data], meta dict
splits, meta = load_splits("zinc_afm")  # {"train", "val", "test"} -> list[Data]
assert not verify()  # checks data/processed against manifest.json
```

Download: `uv run python -m qgf.data.download` (add `--files qgf_raw.tar` for the counts,
`--files qgf_results.tar` for the reference results).

## Licence

Pasqal open-source licence (MIT-derived), see `LICENSE.md`.
