# Paper summary (rewritten; NeurIPS 2026, Garrigues, Sokolov, Hecker, Moro, Acheche)

*Neutral-atom quantum features as complementary structural encodings for graph learning*,
NeurIPS 2026. Summary in our own words, for orientation; numbers are in `claims.md`.

## Question

Classical structural encodings (random-walk probabilities such as RRWP/RWSE, Laplacian
eigenvectors, ring membership) give GNNs useful inductive biases but capture specific
statistics of the graph. Can the dynamics of a neutral-atom quantum processor, driven by
a Hamiltonian that encodes the graph, provide a complementary family of descriptors?
Three sub-questions: what structural information do these features carry on their own;
do they help across many downstream architectures, not one favourable backbone; and do
they survive real hardware noise?

## Method

1. **Graph -> register.** One atom per node, placed in 2D so that bonded atoms sit at
   `r_NN = 5.01 um` (strong van der Waals coupling `J_NN = C6 / r_NN^6`) and non-bonded
   atoms are much farther (weak coupling). Only the topology is encoded: atom and bond
   types are not. This requires a unit-disk-like 2D layout: all 188 MUTAG graphs and 95 %
   of ZINC_subset (11 368 graphs, ZINC\*_subset) qualify.
2. **Quench.** From the all-ground state, evolve under the Rydberg Hamiltonian with a
   constant global Rabi frequency and detuning, in two regimes chosen with a thermalisation
   argument:
   - AFM-like: `Omega = 0.3 J_NN`, `delta = 0` (blockade-dominated, neighbouring
     excitations anti-correlated);
   - FM-like: `Omega = 0.5 J_NN`, `delta = 2 delta_inf`, where `delta_inf` puts the
     initial state at infinite effective temperature; neighbours become positively
     correlated.
3. **Measure.** At 20 times in [0.1, 4] us, 1000 projective shots each, estimate the
   connected occupation correlations `C_ij(t) = <n_i n_j> - <n_i><n_j>`. Diagonal
   `C_ii = n_i (1 - n_i)` -> node features (dimension 20); off-diagonal -> pair features.
   Compressed variants keep 4 statistics over time (min, max, mean, std): AFM_stat, FM_stat.
   Shot count is justified by a CLT bound with a Bonferroni correction over all N(N+1)/2
   entries.
4. **Use.** Features are z-scored with training statistics and injected as node features
   and edge/pair attributes (dense attention biases in the custom graph transformer),
   alone or concatenated with RRWP ("hybrids").

Emulation uses exact state vectors up to 24 atoms and MPS above; ZINC needed about 800
A100 GPU-days. The hardware run executed the AFM pipeline on a neutral-atom QPU for all
188 MUTAG graphs at four times chosen by PCA (about 7.5 QPU-days), with registers and
durations rescaled for the device's lower maximum amplitude and a readout-corrected
estimator.

## Physics intuition

At very short times the correlations are polynomials in the coupling matrix, hence linear
in adjacency powers: classical walk features reproduce them exactly. At the measured
times (hundreds of inverse couplings) the many-body dynamics mixes interference along
cycles, blockade constraints and longer-range propagation, which is no longer a simple
function of walk counts. AFM dynamics is more sensitive to local frustration and
cycles; FM dynamics spreads correlations more coherently across the graph. Time
averages are close to thermal values, so much of the information sits in a static
component, plus oscillating components that matter for pair features.

## Experiments and results

- **Intrinsic benchmark** (15 tasks: node degree, betweenness, eccentricity, bridge
  nodes, smallest cycle; link prediction, edge betweenness, bridge edges, local edge
  connectivity, smallest edge cycle; number of cycles, diameter, density, Wiener index,
  algebraic connectivity). A DeepSet probe without message passing reads each feature set
  alone; 5-fold CV. On ZINC\*_subset, AFM alone already beats RRWP on degree, and
  RRWP+FM+AFM is best or tied-best on most non-saturated tasks. AFM helps most on local
  and cycle tasks, FM on global distance tasks (eccentricity). Compressed statistics lose
  1.1-1.9x in MAE but stay competitive with the Laplacian. MUTAG shows the same trends
  with more noise, and hardware features (HW-AFM) remain informative.
- **Strengthened classical baselines.** Adding ring descriptors and the register geometry
  itself (distances r and couplings J) makes the classical baseline much stronger; the
  quantum features still win on betweenness and bridge nodes, and tie elsewhere. Downstream on MUTAG (GT), RRWP+Laplacian+Rings+geometry is improved by adding
  AFM+FM.
- **Reconstruction ceiling.** Pooled regressions of `C_ij(t)` on classical pair
  descriptors. Initial basis (walk powers, RRWP, Laplacian; OLS, kernel ridge, random
  forest): linear R^2 saturates around 0.83 (MUTAG) and 0.78 (ZINC). Extended basis
  (adds rings, heat kernel, shortest paths, resistance, register geometry, classical-spin
  dTWA, exact two-atom and mean-field responses): about 8 % (AFM) and 14 % (FM) of the
  variance remains unexplained on MUTAG.
- **Residual analysis.** The residual (QGF minus its classical reconstruction) is
  broadband in the graph-Laplacian spectrum for AFM and low-frequency for FM; it decays
  slowly with graph distance, is not explained by classical dynamics (dTWA), and keeps
  most of the graph-level and a large part of node-level information of the full
  features in the intrinsic benchmark (less for edges).
- **Downstream** (GT, GCN, GAT, GINE, Transformer, GeneralConv; MUTAG macro-F1, 10 folds
  x 5 seeds; ZINC\*_subset MAE, 10 seeds; parameter counts matched, no per-feature
  tuning). Quantum features alone do not replace RRWP, but the column best is always a
  quantum-RRWP hybrid. With champions fixed in advance (RRWP vs RRWP+AFM_stat+FM_stat),
  the hybrid ties or wins in 10 of 12 cells (one-sided sign test p = 0.019); Bayesian
  analyses say the effect is positive in direction but small, mostly within practical
  equivalence.
- **Hardware.** QPU and emulated features agree well in rank (per-graph Spearman median
  0.92 one-body, 0.87 two-body). Downstream, HW-AFM_stat is about 0.03 F1 below the
  emulated AFM_stat (0.778 vs 0.806) and RRWP+HW-AFM_stat matches RRWP+AFM_stat within
  about 0.01 F1 (0.849 vs 0.854).
- **ESA with gated fusion** (MUTAG). Plain concatenation to a strong edge-set-attention
  model with RWSE+rings is not consistently useful; a learned gate improves nearly all
  hybrids, best RWSE*+AFM+FM 0.922. The gates attenuate the full time series more than
  the statistics.
- **Placement sensitivity.** Features depend on the specific layout: hardware-level
  position jitter (2 % of the minimum distance) keeps a feature similarity around 0.82,
  5 % drops it to about 0.6, an alternative valid placement gives about 0.57, a node
  permutation gives 0.
- **Also in the paper**: synthetic ladder graphs contrasting pinned and frustrated
  configurations; an exact non-unit-disk graph Hamiltonian on MUTAG (couplings set
  directly from the graph rather than from a 2D layout); one-/two-body and model-size
  ablations on a 1 500-graph ZINC subset; the physics appendix (exact diagonalisation,
  (Omega, delta) sweeps, thermalisation).

## Limitations (as stated or implied)

- No theory: no expressivity (WL) result, no hardness claim; the reconstruction result
  is an empirical ceiling for the tested descriptor families.
- Downstream gains are small, mostly within cross-fold noise; ZINC\*_subset is a
  non-standard subset, not comparable to published ZINC numbers.
- Needs a 2D unit-disk-like embedding (5 % of ZINC dropped) and the features depend on the
  chosen layout.
- Quench parameters come from a heuristic, measurement times are fixed and task-agnostic.
- Cost: emulation and QPU time are far too high for practical use today; hardware run
  covers one regime (AFM), one dataset (MUTAG), four times.
