"""HER catalyst screening service.

Single source of truth used by the CLI (``scripts/14_screen.py``) and the API
(``app/api.py``). Models are loaded lazily and cached so repeated requests don't
re-fit the ETR or reload the MACE checkpoint.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ase.formula import Formula

logger = logging.getLogger(__name__)

ModelName = Literal["etr_emb", "stagea", "ensemble"]

SQLITE_PATH = Path("data/metadata.sqlite")
MACE_DIR = Path("data/mace_features")
SPLITS_PATH = Path("data/splits.json")
STAGEA_CKPT_PATTERN = "logs/checkpoints/mace_ft_stageA_v2_seed*/last.ckpt"
MODEL_CACHE_DIR = Path("data/model_cache")
ETR_CACHE_PATH = MODEL_CACHE_DIR / "etr_emb.pkl"


@lru_cache(maxsize=1)
def load_full_dataset() -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        return pd.read_sql_query(
            "SELECT id, composition, chemical_formula, facet, site_type, coverage, "
            "delta_G_H FROM structures",
            conn,
        )
    finally:
        conn.close()


@lru_cache(maxsize=1)
def load_canonical_test_ids() -> frozenset[str]:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} missing — run `make graphs` first."
        )
    return frozenset(map(str, json.loads(SPLITS_PATH.read_text())["test"]))


@lru_cache(maxsize=1)
def load_all_embeddings() -> dict[str, np.ndarray]:
    """Merge train+val+test MACE embedding npz files into {id: vector}."""
    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        d = np.load(MACE_DIR / f"{split}_emb.npz", allow_pickle=True)
        for sid, x in zip(d["ids"], d["X"], strict=True):
            out[str(sid)] = x.astype(np.float64)
    return out


@lru_cache(maxsize=1)
def available_elements() -> list[str]:
    """Sorted list of metal element symbols present in the dataset."""
    df = load_full_dataset()
    found: set[str] = set()
    for formula in df["chemical_formula"]:
        found.update(parse_metal_elements(formula))
    return sorted(found)


def parse_metal_elements(chemical_formula: str) -> set[str]:
    """Return non-H element symbols from a chemical formula string."""
    return {sym for sym in Formula(chemical_formula).count() if sym != "H"}


def filter_candidates(df: pd.DataFrame, required: set[str],
                       exclude_train: bool = False) -> pd.DataFrame:
    """Filter ``df`` to rows whose metal-element set is a superset of ``required``.

    With ``exclude_train=True``, also restricts to the canonical test set (1172
    IDs) so the ETR cannot return memorised training samples.
    """
    if exclude_train:
        test_ids = load_canonical_test_ids()
        df = df[df["id"].isin(test_ids)]
    mask = df["chemical_formula"].apply(
        lambda f: required.issubset(parse_metal_elements(f))
    )
    return df[mask].reset_index(drop=True)


def _embedding_fingerprint() -> str:
    """SHA-1 of train+val MACE embedding shapes & checksums. Invalidates pickle
    when embeddings change."""
    h = hashlib.sha1()
    for split in ("train", "val"):
        d = np.load(MACE_DIR / f"{split}_emb.npz", allow_pickle=True)
        h.update(str(d["X"].shape).encode())
        h.update(d["X"].tobytes()[:1024]) 
        h.update(str(d["y"].shape).encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _etr_model():
    """Load cached ETR if fingerprint matches; else fit + save. Cached in-process."""
    from sklearn.ensemble import ExtraTreesRegressor

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _embedding_fingerprint()
    if ETR_CACHE_PATH.exists():
        try:
            cached = pickle.loads(ETR_CACHE_PATH.read_bytes())
            if cached.get("fingerprint") == fp:
                logger.info("loaded cached ETR from %s (fingerprint=%s)",
                             ETR_CACHE_PATH, fp)
                return cached["model"]
            logger.info("ETR cache fingerprint stale (%s != %s) — re-fitting",
                         cached.get("fingerprint"), fp)
        except Exception as exc:
            logger.warning("failed to load %s (%s) — re-fitting", ETR_CACHE_PATH, exc)

    parts = []
    for split in ("train", "val"):
        d = np.load(MACE_DIR / f"{split}_emb.npz", allow_pickle=True)
        parts.append((d["X"].astype(np.float64), d["y"].astype(np.float64)))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    logger.info("fitting ETR on %d train+val embeddings (%d dims)", len(X), X.shape[1])
    model = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                 random_state=42, n_jobs=-1)
    model.fit(X, y)
    ETR_CACHE_PATH.write_bytes(pickle.dumps({"fingerprint": fp, "model": model},
                                              protocol=-1))
    logger.info("saved ETR cache to %s", ETR_CACHE_PATH)
    return model


@lru_cache(maxsize=1)
def _stagea_model():
    """Load latest MACE Stage A checkpoint. Cached."""
    import torch

    from models.mace_finetune import LitMACEFineTune

    ckpts = sorted(glob.glob(STAGEA_CKPT_PATTERN))
    if not ckpts:
        raise FileNotFoundError(
            f"no Stage A checkpoint at {STAGEA_CKPT_PATTERN}. "
            "Run `make stagea-multiseed` first."
        )
    ckpt = ckpts[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading Stage A checkpoint: %s (device=%s)", ckpt, device)
    model = LitMACEFineTune.load_from_checkpoint(ckpt, mace_model="medium",
                                                   mace_device=device)
    model.eval()
    if device == "cuda":
        model = model.cuda()
    return model, device


def predict_etr_emb(ids: list[str]) -> np.ndarray:
    emb = load_all_embeddings()
    X = np.vstack([emb[sid] for sid in ids])
    return _etr_model().predict(X)


def predict_stagea(ids: list[str], batch_size: int = 16) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    from data.mace_dataset import MACEDataset

    model, device = _stagea_model()
    ds = MACEDataset(ids=list(ids), z_table=model.z_table, r_max=model.r_max)
    loader = DataLoader(ds, batch_size=batch_size)
    preds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            if device == "cuda":
                batch = batch.cuda()
            preds.extend(model(batch).cpu().tolist())
    return np.array(preds)


def predict(ids: list[str], model: ModelName) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return ΔG_H predictions. For ``ensemble``, returns mean of etr_emb + stagea."""
    if model == "etr_emb":
        return predict_etr_emb(ids)
    if model == "stagea":
        return predict_stagea(ids)
    if model == "ensemble":
        return 0.5 * (predict_etr_emb(ids) + predict_stagea(ids))
    raise ValueError(f"unknown model: {model}")


@dataclass
class ScreenResult:
    elements: list[str]
    model: ModelName
    top: int
    exclude_train: bool
    n_candidates: int
    rows: list[dict]  # one dict per top-N row


def screen(elements: list[str], top: int = 10, model: ModelName = "etr_emb",
            exclude_train: bool = False) -> ScreenResult:
    """Filter dataset by ``elements``, predict ΔG_H, rank by |ΔG_H_pred|."""
    required = {e.capitalize() for e in elements}
    df = load_full_dataset()
    candidates = filter_candidates(df, required, exclude_train=exclude_train)
    if candidates.empty:
        return ScreenResult(elements=sorted(required), model=model, top=top,
                             exclude_train=exclude_train, n_candidates=0, rows=[])

    candidates = candidates.copy()
    ids = candidates["id"].tolist()
    if model == "ensemble":
        p_etr = predict_etr_emb(ids)
        p_sa = predict_stagea(ids)
        candidates["dG_pred_etr"] = p_etr
        candidates["dG_pred_stagea"] = p_sa
        candidates["dG_pred"] = 0.5 * (p_etr + p_sa)
    else:
        candidates["dG_pred"] = predict(ids, model)

    candidates["abs_dG_pred"] = candidates["dG_pred"].abs()
    candidates["error_vs_dft"] = candidates["dG_pred"] - candidates["delta_G_H"]
    top_df = candidates.sort_values("abs_dG_pred").head(top).reset_index(drop=True)

    return ScreenResult(
        elements=sorted(required), model=model, top=top,
        exclude_train=exclude_train, n_candidates=len(candidates),
        rows=top_df.to_dict(orient="records"),
    )
