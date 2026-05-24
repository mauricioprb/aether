"""Adsorption-site geometry shared by filters and features.

The adsorbate is the H atom(s) sitting on the slab. "Central" atoms are the
surface atoms coordinating the adsorbed H (within ``ADS_CUTOFF``); "neighbor"
atoms are the surface atoms coordinating the central atoms (one shell out).
All distances use the minimum-image convention to respect periodicity.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list

ADS_CUTOFF = 2.4  # angstrom, paper's H-surface cutoff (defines central atoms)
COORD_MULT = 1.2  # covalent-radii tolerance for the metal first-neighbor shell


def adsorbate_indices(atoms: Atoms) -> list[int]:
    """Indices of adsorbed H atoms."""
    return [i for i, z in enumerate(atoms.numbers) if z == 1]


def surface_indices(atoms: Atoms) -> list[int]:
    """Indices of non-H (slab) atoms."""
    return [i for i, z in enumerate(atoms.numbers) if z != 1]


def central_indices(atoms: Atoms, cutoff: float = ADS_CUTOFF) -> list[int]:
    """Surface atoms within ``cutoff`` of any adsorbed H."""
    h_idx = adsorbate_indices(atoms)
    surf = surface_indices(atoms)
    if not h_idx or not surf:
        return []
    surf_arr = np.array(surf)
    central: set[int] = set()
    for h in h_idx:
        d = atoms.get_distances(h, surf, mic=True)
        central.update(surf_arr[d <= cutoff].tolist())
    return sorted(central)


def _neighbor_pairs(atoms: Atoms, mult: float = COORD_MULT) -> tuple[np.ndarray, np.ndarray]:
    """All bonded ``(i, j)`` pairs using covalent radii scaled by ``mult``."""
    cutoffs = [c * mult for c in natural_cutoffs(atoms)]
    return neighbor_list("ij", atoms, cutoffs)


def neighbor_indices(atoms: Atoms, central: list[int], mult: float = COORD_MULT) -> list[int]:
    """Slab atoms in the first coordination shell of the central atoms.

    Uses covalent-radii bonding (not the 2.4 A H-cutoff, which is far shorter
    than metal-metal spacing); excludes the central atoms themselves and H.
    """
    if not central:
        return []
    i, j = _neighbor_pairs(atoms)
    central_set = set(central)
    hset = set(adsorbate_indices(atoms))
    neigh = {int(b) for a, b in zip(i, j, strict=True) if a in central_set}
    return sorted(neigh - central_set - hset)


def coordination_numbers(atoms: Atoms, indices: list[int]) -> list[int]:
    """Coordination number (bonded-atom count) for each atom in ``indices``."""
    if not indices:
        return []
    i, _ = _neighbor_pairs(atoms)
    counts = np.bincount(i, minlength=len(atoms))
    return [int(counts[k]) for k in indices]


def h_surface_min_distance(atoms: Atoms) -> float | None:
    """Shortest distance between any adsorbed H and any surface atom (the bond length)."""
    h_idx = adsorbate_indices(atoms)
    surf = surface_indices(atoms)
    if not h_idx or not surf:
        return None
    return float(min(atoms.get_distances(h, surf, mic=True).min() for h in h_idx))


def site_type(n_central: int) -> str:
    """Map coordination count of the adsorbed H to a site label."""
    if n_central <= 1:
        return "top"
    if n_central == 2:
        return "bridge"
    return "hollow"
