"""Dataset filters replicating Wang et al. (2025).

Keep a reaction only if:
  - delta_G_H in [-2, 2] eV
  - H coverage <= 25%
  - shortest H-surface bond in [1.0, 3.0] angstrom (sanity)
The adsorption site (top/bridge/hollow) is assigned from the H coordination
count within 2.4 angstrom.
"""

from __future__ import annotations

import json
import logging

from ase import Atoms

from .geometry import central_indices, h_surface_min_distance, site_type

logger = logging.getLogger(__name__)

DELTA_G_RANGE = (-2.0, 2.0)
COVERAGE_MAX = 0.25
BOND_RANGE = (1.0, 3.0)

HER_EQUATION = "0.5H2(g) + * -> H*"


def is_clean_her(node: dict) -> bool:
    return node.get("Equation") == HER_EQUATION


def passes_metadata_filters(node: dict) -> bool:
    if not is_clean_her(node):
        return False
    dg = node["reactionEnergy"]
    if not (DELTA_G_RANGE[0] <= dg <= DELTA_G_RANGE[1]):
        return False
    return parse_coverage(node) <= COVERAGE_MAX


def parse_coverage(node: dict) -> float:
    raw = node.get("coverages")
    if not raw:
        return 0.0
    try:
        cov = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    values = [float(v) for v in cov.values()]
    return max(values) if values else 0.0


def classify_site(atoms: Atoms) -> tuple[str, int]:
    central = central_indices(atoms)
    return site_type(len(central)), len(central)


def passes_filters(atoms: Atoms, delta_g: float, coverage: float) -> tuple[bool, str | None]:
    if not (DELTA_G_RANGE[0] <= delta_g <= DELTA_G_RANGE[1]):
        return False, "delta_G_H out of [-2, 2]"
    if coverage > COVERAGE_MAX:
        return False, "coverage > 25%"
    bond = h_surface_min_distance(atoms)
    if bond is None:
        return False, "no H-surface bond"
    if not (BOND_RANGE[0] <= bond <= BOND_RANGE[1]):
        return False, f"bond length {bond:.2f} out of [1.0, 3.0]"
    return True, None
