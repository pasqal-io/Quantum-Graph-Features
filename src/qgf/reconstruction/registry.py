"""Population of the descriptor-family registry and named family stacks.

Importing this module registers every basis in ``qgf.reconstruction.design``.
"""

from __future__ import annotations

from qgf.reconstruction import (  # noqa: F401  (imports register the bases)
    features_base,
    features_beyond,
    features_extension,
    features_optional,
    features_ring,
)
from qgf.reconstruction.design import available_bases

#: Base families used as baselines of the extension protocol.
BASELINE_FAMILIES: tuple[str, ...] = (
    "walk",
    "rrwp",
    "walk+laplacian",
    "rrwp+laplacian",
)

#: Atoms of the **classical structural ceiling**, locked after ablation under random
#: forest with graph folds. Excluded (redundant or harmful under the forest, kept as
#: standalone diagnostics): ``geometry``, ``jpow``/``jrrwp``, ``jlaplacian``, ``chemical``.
CLASSICAL_CEILING_ATOMS: tuple[str, ...] = (
    "walk",
    "rrwp",
    "laplacian",
    "ring",
    "heat",
    "geodesic",
    "resistance",
    "interaction",
    "distance",
)
CLASSICAL_CEILING: str = "+".join(CLASSICAL_CEILING_ATOMS)

#: Physical controls added on top of the classical ceiling (extended basis).
PHYSICAL_CONTROL_ATOMS: tuple[str, ...] = ("classical", "twobody", "meanfield")
EXTENDED_BASIS: str = "+".join(CLASSICAL_CEILING_ATOMS + PHYSICAL_CONTROL_ATOMS)

#: The nine cumulative stacks of the extended-ceiling figure, in order.
CUMULATIVE_FAMILIES: tuple[str, ...] = (
    "walk",
    "walk+rrwp",
    "walk+rrwp+laplacian",
    "walk+rrwp+laplacian+ring",
    "walk+rrwp+laplacian+ring+heat+geodesic+resistance",
    CLASSICAL_CEILING,
    CLASSICAL_CEILING + "+classical",
    CLASSICAL_CEILING + "+classical+twobody",
    EXTENDED_BASIS,
)

#: Bases left out of the ceiling but reported as isolated diagnostics.
CLASSICAL_DIAGNOSTICS: tuple[str, ...] = ("geometry", "jpow", "jrrwp", "jlaplacian", "chemical")

#: Beyond-WL stack of the residual-characterisation test (separate from the ceiling).
BEYOND_WL_ATOMS: tuple[str, ...] = ("motif", "symmetry")
BEYOND_WL: str = "+".join(BEYOND_WL_ATOMS)


def extension_families(bases: tuple[str, ...]) -> tuple[str, ...]:
    """Families of the extension protocol: each base alone and with each baseline."""
    families = list(BASELINE_FAMILIES)
    families.extend(bases)
    families.extend(f"{baseline}+{base}" for baseline in BASELINE_FAMILIES for base in bases)
    return tuple(families)


__all__ = [
    "BASELINE_FAMILIES",
    "BEYOND_WL",
    "BEYOND_WL_ATOMS",
    "CLASSICAL_CEILING",
    "CLASSICAL_CEILING_ATOMS",
    "CLASSICAL_DIAGNOSTICS",
    "CUMULATIVE_FAMILIES",
    "EXTENDED_BASIS",
    "PHYSICAL_CONTROL_ATOMS",
    "available_bases",
    "extension_families",
]
