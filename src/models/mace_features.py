from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import Trajectory
from tqdm import tqdm

from data.splits import three_way_split  # noqa: F401  (re-export: canonical split)
from geometry import adsorbate_indices, central_indices
from storage import assign_tags

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mace_E_total",
    "mace_E_per_atom",
    "mace_E_H",
    "mace_E_neighbors_mean",
    "mace_E_neighbors_min",
    "mace_E_surface_mean",
    "mace_n_neighbors",
]


@dataclass
class MACEFeatures:
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, X=self.X, y=self.y,
                 feature_names=np.array(self.feature_names))


def structure_features(atoms, calculator, cutoff_neighbors: float = 2.4) -> np.ndarray:
    atoms = atoms.copy()
    atoms.calc = calculator
    e_total = float(atoms.get_potential_energy())
    e = np.asarray(atoms.calc.results["energies"], dtype=np.float64)

    h_idx = adsorbate_indices(atoms)
    central = central_indices(atoms, cutoff_neighbors)
    surf = np.where(assign_tags(atoms) == 1)[0]

    e_h = float(e[h_idx].mean()) if h_idx else e_total / len(atoms)
    e_nb_mean = float(e[central].mean()) if central else e_h
    e_nb_min = float(e[central].min()) if central else e_h
    e_surf = float(e[surf].mean()) if len(surf) else float(e.mean())

    return np.array([
        e_total, e_total / len(atoms), e_h, e_nb_mean, e_nb_min, e_surf, float(len(central)),
    ], dtype=np.float32)


def extract_features(
    traj_path: str | Path,
    ids: list[str],
    delta_g_h: dict[str, float],
    calculator,
    cutoff_neighbors: float = 2.4,
) -> MACEFeatures:
    frames = {a.info["id"]: a for a in Trajectory(str(traj_path))}
    rows, kept, ys = [], [], []
    for sid in tqdm(ids, desc="MACE features"):
        atoms = frames.get(sid)
        if atoms is None:
            logger.warning("id %s not in traj, skipping", sid)
            continue
        rows.append(structure_features(atoms, calculator, cutoff_neighbors))
        kept.append(sid)
        ys.append(delta_g_h[sid])
    return MACEFeatures(
        ids=np.array(kept, dtype=str),
        X=np.vstack(rows).astype(np.float32),
        y=np.array(ys, dtype=np.float32),
        feature_names=FEATURE_NAMES,
    )


def delta_g_map(traj_path: str | Path) -> dict[str, float]:
    """id -> rotulo de treino (dE_H eletronico; chave legada 'delta_G_H')."""
    return {a.info["id"]: float(a.info["delta_G_H"]) for a in Trajectory(str(traj_path))}


def embedding_names(dim: int) -> list[str]:
    return ([f"embH_{i:03d}" for i in range(dim)]
            + [f"embN_{i:03d}" for i in range(dim)])


def structure_embedding(atoms, calculator, cutoff_neighbors: float = 2.4) -> np.ndarray:
    """Invariant (L=0) MACE node descriptors pooled as [emb(H), mean(emb(neighbors))]."""
    desc = np.asarray(calculator.get_descriptors(atoms, invariants_only=True), dtype=np.float64)
    h_idx = adsorbate_indices(atoms)
    central = central_indices(atoms, cutoff_neighbors)
    emb_h = desc[h_idx].mean(axis=0)
    emb_n = desc[central].mean(axis=0) if central else emb_h
    return np.concatenate([emb_h, emb_n]).astype(np.float32)


def extract_embeddings(
    traj_path: str | Path,
    ids: list[str],
    delta_g_h: dict[str, float],
    calculator,
    cutoff_neighbors: float = 2.4,
) -> MACEFeatures:
    frames = {a.info["id"]: a for a in Trajectory(str(traj_path))}
    rows, kept, ys = [], [], []
    dim = None
    for sid in tqdm(ids, desc="MACE embeddings"):
        atoms = frames.get(sid)
        if atoms is None:
            continue
        vec = structure_embedding(atoms, calculator, cutoff_neighbors)
        dim = dim or len(vec) // 2
        rows.append(vec)
        kept.append(sid)
        ys.append(delta_g_h[sid])
    return MACEFeatures(
        ids=np.array(kept, dtype=str),
        X=np.vstack(rows).astype(np.float32),
        y=np.array(ys, dtype=np.float32),
        feature_names=embedding_names(dim),
    )
