"""The 10 handcrafted descriptors from Wang et al. (2025).

For each adsorption complex we split the slab atoms into:
  - central atoms: surface atoms within 2.4 A of the adsorbed H
  - neighbor atoms: first-shell slab neighbors of the central atoms
and aggregate elemental properties with the geometric mean.

  phi       Nd0**2 / psi0  (composite)
  L_bond    shortest H-central bond length
  Np0       p-electrons, central (geom. mean)
  Nd1       d-electrons, neighbors (geom. mean)
  Out_e0    valence electrons, central (geom. mean)
  R0        atomic radius, central (geom. mean)
  First_IE0 first ionization energy, central (geom. mean)
  CN        coordination number of central atoms (geom. mean)
  Out_e1    valence electrons, neighbors (geom. mean)
  psi1      Pauling electronegativity, neighbors (geom. mean)

Property choices (documented deviations where the paper is ambiguous): Nd/Np are
total d/p electron counts; Out_e is mendeleev ``nvalence`` (group-consistent,
unlike the literal outer shell which is anomalous for e.g. Pd); CN uses a
covalent-radii neighbor list.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import lru_cache

import numpy as np
from ase import Atoms
from mendeleev import element

from geometry import (
    adsorbate_indices,
    central_indices,
    coordination_numbers,
    neighbor_indices,
)

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "phi", "L_bond", "Np0", "Nd1", "Out_e0",
    "R0", "First_IE0", "CN", "Out_e1", "psi1",
]


@lru_cache(maxsize=256)
def atom_props(symbol: str) -> dict[str, float]:
    e = element(symbol)
    by_l: dict[str, int] = defaultdict(int)
    for (_, sub), count in e.ec.conf.items():
        by_l[sub] += count
    radius = e.atomic_radius or e.covalent_radius
    return {
        "n_d": float(by_l.get("d", 0)),
        "n_p": float(by_l.get("p", 0)),
        "out_e": float(e.nvalence()),
        "radius": float(radius),
        "ie1": float(e.ionenergies.get(1)),
        "en": float(e.en_pauling) if e.en_pauling else 0.0,
    }


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.any(arr == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(np.abs(arr)))))


def _prop_geomean(atoms: Atoms, indices: list[int], key: str) -> float:
    syms = [atoms[i].symbol for i in indices]
    return geometric_mean([atom_props(s)[key] for s in syms])


def compute_features(atoms: Atoms) -> dict[str, float] | None:
    central = central_indices(atoms)
    if not central:
        logger.debug("no central atoms for %s", atoms.info.get("id"))
        return None
    neighbors = neighbor_indices(atoms, central)
    h_idx = adsorbate_indices(atoms)

    l_bond = min(
        float(atoms.get_distances(h, central, mic=True).min()) for h in h_idx
    )

    nd0 = _prop_geomean(atoms, central, "n_d")
    psi0 = _prop_geomean(atoms, central, "en")
    phi = (nd0 ** 2) / psi0 if psi0 else 0.0

    cn = geometric_mean([float(c) for c in coordination_numbers(atoms, central)])

    feats = {
        "phi": phi,
        "L_bond": l_bond,
        "Np0": _prop_geomean(atoms, central, "n_p"),
        "Nd1": _prop_geomean(atoms, neighbors, "n_d"),
        "Out_e0": _prop_geomean(atoms, central, "out_e"),
        "R0": _prop_geomean(atoms, central, "radius"),
        "First_IE0": _prop_geomean(atoms, central, "ie1"),
        "CN": cn,
        "Out_e1": _prop_geomean(atoms, neighbors, "out_e"),
        "psi1": _prop_geomean(atoms, neighbors, "en"),
    }
    if any(np.isnan(v) for v in feats.values()):
        return None
    return feats
