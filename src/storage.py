"""Persist the curated dataset in three aligned representations.

A "record" is a dict with keys: ``id, atoms, delta_G_H, coverage, site_type,
composition, chemical_formula, facet, n_atoms, source, features`` where
``features`` is the 10-descriptor dict (or None). The ``.traj`` frame order,
the SQLite ``traj_index``/``lmdb_key`` and the LMDB integer key all coincide.
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.io import Trajectory

from features import FEATURE_NAMES

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS structures (
  id TEXT PRIMARY KEY,
  composition TEXT,
  chemical_formula TEXT,
  facet TEXT,
  site_type TEXT,
  coverage REAL,
  delta_G_H REAL,
  n_atoms INTEGER,
  source TEXT,
  traj_index INTEGER,
  lmdb_key INTEGER
);
CREATE TABLE IF NOT EXISTS features (
  id TEXT PRIMARY KEY,
  phi REAL, L_bond REAL, Np0 REAL, Nd1 REAL,
  Out_e0 REAL, R0 REAL, First_IE0 REAL,
  CN REAL, Out_e1 REAL, psi1 REAL,
  FOREIGN KEY (id) REFERENCES structures(id)
);
CREATE INDEX IF NOT EXISTS idx_composition ON structures(composition);
CREATE INDEX IF NOT EXISTS idx_site_type ON structures(site_type);
"""


def assign_tags(atoms: Atoms, layer_thresh: float = 1.0) -> np.ndarray:
    """fairchem-style tags: 2=adsorbate (H), 1=surface layer, 0=sub-surface."""
    z = atoms.positions[:, 2]
    is_h = atoms.numbers == 1
    tags = np.zeros(len(atoms), dtype=np.int64)
    tags[is_h] = 2
    metal = ~is_h
    if metal.any():
        top_z = z[metal].max()
        tags[metal & (z >= top_z - layer_thresh)] = 1
    return tags


def write_traj(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Single .traj with all frames; metadata in each ``atoms.info``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(str(path), "w") as traj:
        for rec in records:
            atoms = rec["atoms"]
            atoms.info = {
                "id": rec["id"],
                "delta_G_H": rec["delta_G_H"],
                "source": rec["source"],
                "site_type": rec["site_type"],
                "composition": rec["composition"],
            }
            traj.write(atoms)
    logger.info("wrote %d frames to %s", len(records), path)
    return path


def _to_data_dict(atoms: Atoms, y: float, sid: str) -> dict[str, Any]:
    """fairchem-compatible payload (torch tensors), loadable into a Data object."""
    tags = assign_tags(atoms)
    return {
        "pos": torch.tensor(atoms.positions, dtype=torch.float),
        "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long),
        "cell": torch.tensor(np.array(atoms.cell), dtype=torch.float).unsqueeze(0),
        "natoms": len(atoms),
        "tags": torch.tensor(tags, dtype=torch.long),
        "y": float(y),
        "sid": sid,
    }


def write_lmdb(records: list[dict[str, Any]], path: str | Path,
               map_size: int = 1 << 34) -> Path:
    """LMDB keyed by ascii integer index, value = pickled fairchem-style dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=map_size, subdir=False, meminit=False)
    with env.begin(write=True) as txn:
        for idx, rec in enumerate(records):
            data = _to_data_dict(rec["atoms"], rec["delta_G_H"], rec["id"])
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.put(b"length", pickle.dumps(len(records), protocol=-1))
    env.sync()
    env.close()
    logger.info("wrote %d entries to %s", len(records), path)
    return path


def write_sqlite(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Populate ``structures`` and ``features`` tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            """INSERT OR REPLACE INTO structures
               (id, composition, chemical_formula, facet, site_type, coverage,
                delta_G_H, n_atoms, source, traj_index, lmdb_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (rec["id"], rec["composition"], rec["chemical_formula"], rec["facet"],
                 rec["site_type"], rec["coverage"], rec["delta_G_H"], rec["n_atoms"],
                 rec["source"], idx, idx)
                for idx, rec in enumerate(records)
            ],
        )
        conn.executemany(
            f"""INSERT OR REPLACE INTO features
                (id, {', '.join(FEATURE_NAMES)})
                VALUES (?, {', '.join('?' * len(FEATURE_NAMES))})""",
            [
                (rec["id"], *[rec["features"][f] for f in FEATURE_NAMES])
                for rec in records if rec.get("features")
            ],
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("wrote %d structures to %s", len(records), path)
    return path


def load_features_frame(path: str | Path):
    """Load joined structures+features as a DataFrame for the baseline."""
    import pandas as pd

    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query(
            """SELECT s.id, s.site_type, s.composition, s.delta_G_H,
                      f.phi, f.L_bond, f.Np0, f.Nd1, f.Out_e0, f.R0,
                      f.First_IE0, f.CN, f.Out_e1, f.psi1
               FROM structures s JOIN features f ON s.id = f.id""",
            conn,
        )
    finally:
        conn.close()
