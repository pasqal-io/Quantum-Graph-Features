# Temporal PCA of the quantum features (App. D.2, Fig. 4)

```bash
uv run python experiments/temporal_pca/run.py --dataset zinc    # Fig. 4 (about 1 min on CPU)
uv run python experiments/temporal_pca/run.py --dataset mutag
```

Outputs `results/temporal_pca/pca_<dataset>.{json,png}`. Features are z-scored per time
step over the whole dataset (as in training), rows = nodes (one-body) or directed edges
(two-body), PCA over the 20 times. Matches the original script to <= 1.1e-5 in every
explained-variance ratio (`scripts/internal/generation_reference_pca.py`).

| dataset, pulse | one-body PC1 | 90% / 95% | two-body PC1 (PC2, PC3) | 90% / 95% |
|---|---|---|---|---|
| ZINC*_subset AFM | 80.6% | 5 / 9 | 59.4% (9.2, 7.1) | 9 / 14 |
| ZINC*_subset FM | 80.9% | 2 / 5 | 76.9% (4.8, 2.1) | 8 / 14 |
| MUTAG AFM | 82.5% | 4 / 10 | 64.6% (8.4, 5.6) | 10 / 14 |
| MUTAG FM | 87.4% | 2 / 5 | 71.9% (5.0, 3.0) | 10 / 15 |

The text of App. D.2 ("two-body PC1 59-65%", "90% needs 4-5 one-body components") holds
for AFM; FM is more concentrated (two-body PC1 72-77%, 2 one-body components for 90%).
