"""Adsorption-site geometry shared by filters and features.

The adsorbate is the H atom(s) sitting on the slab. "Central" atoms are the
surface atoms coordinating the adsorbed H (within ``ADS_CUTOFF``); "neighbor"
atoms are the surface atoms coordinating the central atoms (one shell out).
All distances use the minimum-image convention to respect periodicity.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import natural_cutoffs, neighbor_list

ADS_CUTOFF = 2.4  # angstrom, paper's H-surface cutoff (defines central atoms)
COORD_MULT = 1.2  # covalent-radii tolerance for the metal first-neighbor shell

# Atomos mais proximos que MIN_DISTANCE_RATIO x (raio covalente somado) estao
# sobrepostos. Calibrado, nao chutado: nas 7238 estruturas do dataset o menor
# valor observado e 0.669, entao 0.60 nao rejeita nada legitimo e ainda pega as
# geometrias em que o modelo extrapola sem avisar (H a 0.8 A de um Pt ontop da
# 0.479 e uma predicao de +13.6 eV com cara de resposta).
MIN_DISTANCE_RATIO = 0.60


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


def min_distance_ratio(atoms: Atoms) -> float:
    """Menor distancia interatomica em unidades de raio covalente somado.

    1.0 significa dois atomos exatamente encostados pelos raios covalentes;
    abaixo de ``MIN_DISTANCE_RATIO`` eles se sobrepoem e a estrutura nao e
    fisicamente plausivel.
    """
    if len(atoms) < 2:
        return float("inf")
    # ponytail: O(n^2) com a matriz inteira; o /predict limita a 512 atomos, o
    # que da 262k pares e roda em milissegundos. Se o teto subir, trocar por
    # neighbor_list com cutoff.
    d = atoms.get_all_distances(mic=True)
    radii = covalent_radii[atoms.numbers]
    np.fill_diagonal(d, np.inf)
    return float((d / (radii[:, None] + radii[None, :])).min())
