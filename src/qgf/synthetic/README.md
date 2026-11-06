# Synthetic graphs

**Ladder graphs, pinned vs frustrated** (done): ladders with crossings whose blockade
ground state is unique (pinned) or degenerate (frustrated), grouped so that pooled RRWP-2
features are identical inside each group. Experiment, commands and expected numbers:
[`experiments/ladder/README.md`](../../../experiments/ladder/README.md).

| module | content |
|---|---|
| `ladder.py` | ladder builder, exact ground state (rung transfer-matrix DP, any degeneracy), brute-force reference, unit-disk atom register |
| `blind.py` | RRWP-K signatures, blind groups, group-balanced subset |
| `features.py` | pooled RRWP / Laplacian / ring / geometry / quench / ground-state families |
| `dataset.py` | qgf datasets `ladder_afm` / `ladder_fm` and their benchmark inputs |
| `benchmark.py` | logistic regression on the stored folds, accuracy / F1 / ROC-AUC, permutation test |

**Exact non-unit-disk graph Hamiltonian on MUTAG** (couplings set directly from the graph
instead of a 2D register): `experiments/non_unit_disk/`.
