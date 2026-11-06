# Glossary

- **QGF (quantum graph feature)**: connected occupation correlation `C_ij(t)` measured
  after a quench of the graph-encoded Rydberg Hamiltonian; diagonal = node feature,
  off-diagonal = pair feature, 20 times (4 on the QPU).
- **Rydberg Hamiltonian**: `H = sum_{i<j} J_ij n_i n_j + sum_i (Omega/2 sigma^x_i - delta n_i)`,
  `J_ij = C6 / r_ij^6`, `C6/hbar = 8.66e5 um^6 rad/us` (Rydberg level 60).
- **J_NN, r_NN**: target nearest-neighbour coupling and distance (`r_NN = 5.01 um`,
  `J_NN/hbar ~ 54.75 rad/us`); every embedding has minimum distance r_NN.
- **Graph embedding**: 2D atom positions (`pos`, um) such that edges are at ~r_NN and
  non-edges far. Shipped for MUTAG and ZINC; new graphs: Blade (`qoolqit`).
- **Unit-disk graph**: graph realisable by points with edges exactly between points closer
  than a radius; the encoding needs graphs close to this.
- **ZINC\*_subset**: the 11 368 of the 12 000 ZINC_subset graphs with a good 2D embedding
  (embedding-quality criterion on edge-length uniformity ELU and gap ratio GR); splits
  9475 / 946 / 947. **ZINC1500**: ~1 500-graph subset used in the ablations.
- **AFM quench**: `Omega = 0.3 J_NN`, `delta = 0`; anti-correlated neighbours, sensitive
  to local frustration and cycles.
- **FM quench**: `Omega = 0.5 J_NN`, `delta = 2 delta_inf`, `delta_inf = sum_{i!=j} J_ij / (4N)`
  (infinite effective temperature of the initial state); positively correlated
  neighbours, more global.
- **HW-AFM**: AFM features measured on the QPU (MUTAG, 4 times). The one-body
  diagonal uses the single-atom readout correction (`mutag_hw_afm`).
- **AFM_stat / FM_stat**: 4 statistics over time (min, max, mean, std). **_mean**: time
  mean only.
- **ZZ convention**: correlations stored as `4 C` (= connected `<Z_i Z_j>`), so values lie
  in [-1, 1]; `meta["zz_scale_applied"]` records the factor applied at export.
- **Time-major layout**: `onebody_features [T, N]`, `twobody_features [T, N, N]`.
- **Readout-corrected jackknife**: QPU estimator correcting false positives/negatives
  (`eps = 0.025`, `eps' = 0.1`) with jackknife errors.
- **Shot budget (CLT-Bonferroni)**: `shots(N) = ceil(v_max k_N^2 / eps^2)`,
  `k_N = Phi^-1(1 - delta/(N(N+1)))`, `v_max = 0.06`, `eps = 0.03`.
- **RRWP**: relative random-walk probabilities, powers of `D^-1 A` up to k = 20 (matches
  T = 20). **RWSE**: their diagonal. **RWSE\***: RWSE + ring membership (ESA only).
- **Laplacian PE**: 4 eigenvectors of the normalised Laplacian (+ outer products as pair
  features in the GT).
- **Rings**: ring-membership descriptors (minimum cycle basis, sizes 3-8).
- **Geometry (r, J)**: register distances and couplings used as classical features; the
  strongest classical baseline, since the register is the encoding.
- **Random / No data controls**: Gaussian pseudo-features of matched dimension; bare
  molecular attributes.
- **DeepSet probe**: `rho(phi(x))`, 2-layer width-64 MLPs; per-entity for node/edge tasks,
  sum-pooled for graph tasks; no message passing, so it measures what the features
  themselves encode.
- **Intrinsic benchmark**: 15 structural tasks probed by the DeepSet, 5-fold CV.
- **Hybrid**: classical encoding + quantum features (e.g. RRWP+AFM+FM).
- **Fixed champions**: RRWP (classical) vs RRWP+AFM_stat+FM_stat (hybrid), chosen by
  average rank across the sweep before the significance tests.
- **ROPE**: region of practical equivalence in the Bayesian tests.
- **Reconstruction ceiling**: best R^2 of a pooled regression (one coefficient vector for
  all pairs of all graphs) of `C_ij(t)` on classical pair descriptors; `1 - R^2` = part of
  the dynamics outside the span of those descriptors. **Extended basis**: walk, RRWP,
  Laplacian, rings, heat kernel, shortest path, resistance, geometry, dTWA, exact two-atom
  and mean-field responses.
- **Residual**: QGF minus its classical (OLS, extended ceiling) reconstruction; released
  as `*_residual` datasets.
- **dTWA**: discrete truncated Wigner approximation, a classical-spin simulation of the
  dynamics, used as a classical control.
- **Retention**: share of the full features' intrinsic-benchmark gain over Random kept by
  the residual.
- **Placement sensitivity**: Spearman similarity of pair features between a reference and
  a perturbed or re-embedded layout.
- **GT**: the custom dense graph transformer with GRIT-style additive attention biases;
  **ESA**: edge-set attention; **gated fusion**: learned gate per feature stream.
