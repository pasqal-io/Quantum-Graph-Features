# Other GNN backbones (Tables 2-3, geometry table)

PyTorch Geometric backbones of Tables 2-3: GCN, GAT, GINE, Transformer (`TransformerConv`)
and GeneralConv, with the folds, seeds and encodings of the GT benchmark
(`src/qgf/downstream/gt/`). Experiment, commands and expected numbers:
[`experiments/downstream_gnns/README.md`](../../../../experiments/downstream_gnns/README.md).

| module | content |
|---|---|
| `model.py` | `GNNWithPE`: atom / bond embeddings, one-body encodings concatenated to the nodes, two-body encodings as edge attributes (or one learned edge weight for GCN), mean pooling, MLP head |
| `benchmark.py` | one trial: data and encodings (shared with the GT port), training loop with early stopping on the validation split, `results.json` |

The one-/two-body and model-size ablations (Tables 8-10, `zinc_tiny_{afm,fm}`) reuse this
model: `experiments/ablations/`.
