# Statistical analysis — downstream feature-set sweep

## Champions (artefact 5a)

Classical champion (best avg across 12 cells): **rrwp**
Quantum champion (best avg across 12 cells):   **rrwp_afm_stat_fm_stat**

---

## ROPE choice

**Cell-level test (6a):** rope = ±0.5 in normalised units
(each paired mean diff is divided by $\sqrt{\sigma_{\text{cl}}^2 + \sigma_{\text{qu}}^2}$).

**Trial-level test (6b):** rope = ±0.01 F1 points.
Median within-cell std across MUTAG cells for the two champion configs is **0.0774 F1**.
The rope (0.01) is ~0.13× the typical cell noise.

---

## Artefact 5b — Champion vs champion frequentist sign test

H₁: quantum (`rrwp_afm_stat_fm_stat`) better than classical (`rrwp`)
Sign convention: positive Δ = quantum better (MUTAG: ΔF1; ZINC: ΔMAE flipped).

**k = 10 wins out of n = 12 cells → p = 0.0193**

| arch | dataset | Δ (sign-corrected) | within noise? |
|---|---|---|---|
| GT | MUTAG | 0.0074 | ✓ |
| GT | ZINC_small | 0.0074 |  |
| GCN | MUTAG | 0.0112 | ✓ |
| GCN | ZINC_small | 0.0050 |  |
| GAT | MUTAG | 0.0058 | ✓ |
| GAT | ZINC_small | 0.0140 |  |
| GINE | MUTAG | 0.0018 | ✓ |
| GINE | ZINC_small | 0.0010 | ✓ |
| Transformer | MUTAG | 0.0021 | ✓ |
| Transformer | ZINC_small | -0.0020 | ✓ |
| GeneralConv | MUTAG | 0.0018 | ✓ |
| GeneralConv | ZINC_small | -0.0010 | ✓ |

---

## Artefact 5c — Best-of-class per cell sign test

For each (arch, dataset) cell the best classical and best hybrid config
are selected independently (best mean; ties → lower std).

**k = 11 wins out of n = 11 non-tied cells → p = 4.88e-04**

| arch | dataset | best classical | best hybrid | Δ | tie? |
|---|---|---|---|---|---|
| GT | MUTAG | rrwp_laplacian | rrwp_afm_fm | 0.0146 |  |
| GT | ZINC_small | rrwp | rrwp_afm_stat | 0.0076 |  |
| GCN | MUTAG | rrwp_laplacian | rrwp_afm | 0.0272 |  |
| GCN | ZINC_small | rrwp | rrwp_afm_stat | 0.0080 |  |
| GAT | MUTAG | rrwp | rrwp_fm_stat | 0.0138 |  |
| GAT | ZINC_small | rrwp | rrwp_afm_stat_fm_stat | 0.0140 |  |
| GINE | MUTAG | rrwp | rrwp_afm_fm | 0.0063 |  |
| GINE | ZINC_small | rrwp | rrwp_afm_stat_fm_stat | 0.0010 |  |
| Transformer | MUTAG | rrwp | rrwp_afm_fm | 0.0086 |  |
| Transformer | ZINC_small | rrwp_laplacian | rrwp_fm_stat | 0.0030 |  |
| GeneralConv | MUTAG | rrwp | rrwp_fm_stat | 0.0045 |  |
| GeneralConv | ZINC_small | rrwp_laplacian | rrwp_fm_stat | 0.0000 | ✓ |

---

## Artefact 6a — Bayesian sign test (cell-level)

Input: 12 normalised cell-level diffs; rope = ±0.5.
Convention: baycomp P(left) = P(first arg better); first arg = quantum (diffs\_norm > 0).

| | value | 95% CI |
|---|---|---|
| P(quantum clearly better)   | 0.0194 | [0.0000, 0.3888] |
| P(rope / equiv)             | 0.9806 | [0.6112, 1.0000] |
| P(classical clearly better) | 0.0000 | [0.0000, 0.0000] |

Plot: `bayes_sign_test_cell.pdf` + `.png` (saved: True)

---

## Artefact 6b — Bayesian signed-rank test (MUTAG trial-level)

Input: 300 paired F1 observations (GT: 50, GCN: 50, GAT: 50, GINE: 50, Transformer: 50, GeneralConv: 50).
Rope = ±0.01 F1 pts. Median within-cell std = 0.0774.
Convention: baycomp P(left) = P(first arg better); first arg = quantum F1.

| | value | 95% CI |
|---|---|---|
| P(quantum better)   | 0.9067 | [0.2510, 1.0000] |
| P(rope / equiv)     | 0.0000 | [0.0000, 0.0000] |
| P(classical better) | 0.0933 | [0.0000, 0.7490] |

Plot: `bayes_signedrank_mutag.pdf` + `.png` (saved: True)

---

### 6(c) Hierarchical test (Corani-Benavoli 2017)

Convention: P(left) = P(quantum better), P(right) = P(classical better).

| | value | 95% CI |
|---|---|---|
| P(quantum better)   | 0.2397 | [0.0000, 0.4557] |
| P(rope / equiv)     | 0.7270 | [0.0494, 1.0000] |
| P(classical better) | 0.0332 | [0.0000, 0.9505] |
| architectures       | 6 | |
| pairs per arch      | 50 | |
| rope                | ±0.01 F1 | |

Plot: `bayes_hierarchical_mutag.pdf` + `.png` (saved: True)

---

## Extended sensitivity analysis

### Analysis 1 — Cell-level sign test (sensitivity over ROPE, Cohen's d)

| ROPE | P(hybrid) | P(rope) | P(classical) |
|---|---|---|---|
| ±0.2 | 0.0319 | 0.9665 | 0.0016 |
| ±0.5 | 0.0191 | 0.9809 | 0.0000 | ←
| ±0.8 | 0.0030 | 0.9970 | 0.0000 |

**HDI 95% on mean Cohen's d:** median = 0.3674, [-0.2145, 0.9524]

### Analysis 2 — Trial-level signed-rank (sensitivity over ROPE, F1)

| ROPE | P(hybrid) | P(rope) | P(classical) |
|---|---|---|---|
| ±0.005 | 0.8510 | 0.0000 | 0.1490 |
| ±0.01 | 0.9084 | 0.0000 | 0.0916 | ←
| ±0.02 | 0.9077 | 0.0056 | 0.0867 |

**HDI 95% on mean diff (quantum − classical):** median = 0.0050 F1, [-0.0028, 0.0128] F1

### Analysis 3 — Hierarchical test (sensitivity over ROPE, F1)

| ROPE | P(hybrid) | P(rope) | P(classical) |
|---|---|---|---|
| ±0.005 | 0.5722 | 0.3151 | 0.1127 |
| ±0.01 | 0.2397 | 0.7270 | 0.0332 | ←
| ±0.02 | 0.0210 | 0.9744 | 0.0045 |

**HDI 95% on global μ (quantum − classical):** median = 0.0051 F1, [-0.0088, 0.0187] F1
**HDI 95% on τ (between-arch std):** median = 0.0055 F1, [0.0008, 0.0253] F1

**Per-architecture 95% interval (independent t-posterior):**

| Architecture | median F1 diff | lo | hi |
|---|---|---|---|
| GT | 0.0074 | -0.0149 | 0.0298 |
| GCN | 0.0112 | -0.0078 | 0.0301 |
| GAT | 0.0058 | -0.0147 | 0.0262 |
| GINE | 0.0019 | -0.0193 | 0.0228 |
| Transformer | 0.0021 | -0.0163 | 0.0206 |
| GeneralConv | 0.0018 | -0.0141 | 0.0177 |

Forest plot: `bayes_hierarchical_forest.pdf` + `.png` (saved: True)
