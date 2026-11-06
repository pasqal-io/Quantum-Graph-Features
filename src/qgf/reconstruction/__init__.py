"""Classical reconstruction ceiling of quantum graph features and residual analysis.

Pooled regression of the quantum correlations C_ij(t) on classical pair descriptors. The
R^2 is an expressivity ceiling, not a learning score: 1 - R^2 is the part of the quantum
dynamics outside the span of the tested descriptor family.
"""

from qgf.reconstruction import registry  # noqa: F401  (registers every basis)
