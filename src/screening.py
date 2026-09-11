"""HER catalyst screening service.

Single source of truth used by the CLI (``scripts/14_screen.py``) and the API
(``app/api.py``). Models are loaded lazily and cached so repeated requests don't
re-fit the ETR or reload the MACE checkpoint.
"""

from __future__ import annotations

import glob
import hashlib
import hmac
import json
import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ase import Atoms
from ase.formula import Formula

logger = logging.getLogger(__name__)

ModelName = Literal["etr_emb", "stagea", "ensemble"]

# dG_H = dE_H + 0.24 eV (Norskov 2005). O rotulo depositado no Catalysis Hub
# e a energia ELETRONICA dE_H - verificado em 2026-07-02: (1) cathub/organize.py
# computa reactionEnergy por diferenca de get_potential_energy() (dE por
# construcao); (2) Mamun 2019, Eq. (1), define E_ads em energias totais DFT sem
# ZPE/entropia; (3) H*/Pt(111) no banco = -0.19..-0.31 eV (faixa de dE; como dG
# contradiria o dG_H(Pt) ~ -0.09 eV canonico). Fontes do dataset usam funcionais
# distintos (Mamun: BEEF-vdW; Yohannes 2023: VASP/PBE) - declarar na dissertacao.
SABATIER_CORRECTION_EV = 0.24

SQLITE_PATH = Path("data/metadata.sqlite")
MACE_DIR = Path("data/mace_features")
SPLITS_PATH = Path("data/splits.json")
STAGEA_CKPT_PATTERN = "logs/checkpoints/mace_ft_stageA_v2_seed*/last.ckpt"
MODEL_CACHE_DIR = Path("data/model_cache")
ETR_CACHE_PATH = MODEL_CACHE_DIR / "etr_emb.pkl"

_ETR_CACHE_KEY = os.environ.get("ETR_CACHE_KEY", "").encode()
_HMAC_LEN = 32  # sha256 digest size


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
            f"{SPLITS_PATH} missing - run `make graphs` first."
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


def _load_signed_cache(path: Path) -> dict:
    """Read an HMAC-tagged pickle, verifying authenticity before unpickling.

    Raises if no key is configured or the tag does not match, so a tampered
    cache file can never reach ``pickle.loads``.
    """
    if not _ETR_CACHE_KEY:
        raise ValueError("ETR_CACHE_KEY not set - on-disk cache untrusted")
    raw = path.read_bytes()
    sig, blob = raw[:_HMAC_LEN], raw[_HMAC_LEN:]
    expected = hmac.new(_ETR_CACHE_KEY, blob, "sha256").digest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("ETR cache HMAC mismatch - refusing to unpickle")
    return pickle.loads(blob)


def _save_signed_cache(path: Path, obj: dict) -> None:
    """Write an HMAC-tagged pickle. No-op (warns) when no key is configured."""
    if not _ETR_CACHE_KEY:
        logger.warning("ETR_CACHE_KEY not set - skipping disk cache write")
        return
    blob = pickle.dumps(obj, protocol=-1)
    sig = hmac.new(_ETR_CACHE_KEY, blob, "sha256").digest()
    path.write_bytes(sig + blob)


@lru_cache(maxsize=1)
def _etr_model():
    """Load cached ETR if fingerprint matches; else fit + save. Cached in-process."""
    from sklearn.ensemble import ExtraTreesRegressor

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _embedding_fingerprint()
    if ETR_CACHE_PATH.exists():
        try:
            cached = _load_signed_cache(ETR_CACHE_PATH)
            if cached.get("fingerprint") == fp:
                logger.info("loaded cached ETR from %s (fingerprint=%s)",
                             ETR_CACHE_PATH, fp)
                return cached["model"]
            logger.info("ETR cache fingerprint stale (%s != %s) - re-fitting",
                         cached.get("fingerprint"), fp)
        except Exception as exc:
            logger.warning("failed to load %s (%s) - re-fitting", ETR_CACHE_PATH, exc)

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
    _save_signed_cache(ETR_CACHE_PATH, {"fingerprint": fp, "model": model})
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


def predictable_mask(df: pd.DataFrame, model: ModelName) -> pd.Series:
    """Quais linhas o ``model`` consegue prever.

    ``etr_emb`` (e ``ensemble``, que o contem) le embeddings de um cache que
    cobre so as 5860 estruturas curadas, enquanto o dataset tem 7238. Sem este
    filtro, qualquer triagem que inclua o conjunto de treino estoura KeyError
    dentro do predict e vira 500.
    """
    if model == "stagea":
        return pd.Series(True, index=df.index)
    have = load_all_embeddings().keys()
    return df["id"].isin(have)


@lru_cache(maxsize=1)
def _mp_calculator():
    """MACE-MP-0 pristino, para embutir estrutura que nao esta no dataset.

    Nao da pra reaproveitar o backbone do Stage A: ele foi fine-tunado, e o ETR
    foi treinado sobre descritores do MP-0 intacto. Trocar um pelo outro muda a
    distribuicao das features sem erro nenhum aparecer.
    """
    import torch
    from mace.calculators import mace_mp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading MACE-MP-0 descriptors calculator (device=%s)", device)
    return mace_mp(model="medium", device=device, default_dtype="float32")


def embed_atoms(frames: list[Atoms]) -> np.ndarray:
    """Embedding MACE (512-d) de estruturas arbitrarias, no mesmo pooling do treino."""
    from models.mace_features import structure_embedding

    calc = _mp_calculator()
    return np.vstack([structure_embedding(a, calc) for a in frames]).astype(np.float64)


def predict_etr_emb(ids: list[str]) -> np.ndarray:
    emb = load_all_embeddings()
    X = np.vstack([emb[sid] for sid in ids])
    return _etr_model().predict(X)


def predict_etr_emb_atoms(frames: list[Atoms]) -> np.ndarray:
    return _etr_model().predict(embed_atoms(frames))


def predict_stagea_atoms(frames: list[Atoms], batch_size: int = 16) -> np.ndarray:
    """Stage A sobre estruturas arbitrarias. Caminho unico: o de ids so resolve
    os Atoms antes de chegar aqui."""
    import torch
    from torch_geometric.loader import DataLoader

    from data.mace_dataset import graph_from_atoms

    model, device = _stagea_model()
    graphs = [graph_from_atoms(a, model.z_table, model.r_max) for a in frames]
    preds: list[float] = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size):
            if device == "cuda":
                batch = batch.cuda()
            preds.extend(model(batch).cpu().tolist())
    return np.array(preds)


def predict_stagea(ids: list[str], batch_size: int = 16) -> np.ndarray:
    from data.mace_dataset import TRAJ_PATH, _load_frames

    frames = _load_frames(str(TRAJ_PATH))
    return predict_stagea_atoms([frames[i] for i in ids if i in frames], batch_size)


def predict(ids: list[str], model: ModelName) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return ΔG_H predictions. For ``ensemble``, returns mean of etr_emb + stagea."""
    if model == "etr_emb":
        return predict_etr_emb(ids)
    if model == "stagea":
        return predict_stagea(ids)
    if model == "ensemble":
        return 0.5 * (predict_etr_emb(ids) + predict_stagea(ids))
    raise ValueError(f"unknown model: {model}")


def predict_atoms(frames: list[Atoms], model: ModelName) -> np.ndarray:
    """ΔE_H previsto para estruturas que nao estao no dataset.

    Mesmos pesos do caminho por id; o que muda e so de onde vem o Atoms. Some o
    ``SABATIER_CORRECTION_EV`` por fora para obter ΔG_H, como no /screen.
    """
    if model == "etr_emb":
        return predict_etr_emb_atoms(frames)
    if model == "stagea":
        return predict_stagea_atoms(frames)
    if model == "ensemble":
        return 0.5 * (predict_etr_emb_atoms(frames) + predict_stagea_atoms(frames))
    raise ValueError(f"unknown model: {model}")


@dataclass
class ScreenResult:
    elements: list[str]
    model: ModelName
    top: int
    exclude_train: bool
    n_candidates: int
    dg_correction: float
    rows: list[dict]  # one dict per top-N row
    # ΔG_H previsto de TODAS as candidatas. O modelo ja roda sobre o conjunto
    # inteiro antes do corte, entao sai de graca, e o vulcao precisa das
    # encostas para o apice significar alguma coisa.
    pool_dG_pred: list[float] = field(default_factory=list)


def screen(elements: list[str], top: int = 10, model: ModelName = "etr_emb",
            exclude_train: bool = False,
            dg_correction: float = SABATIER_CORRECTION_EV) -> ScreenResult:
    """Filter dataset by ``elements``, predict, rank by |ΔG_H_pred|.

    Predictions live in label space (ΔE_H); ``dg_correction`` converts to
    ΔG_H for the Sabatier ranking. Pass 0.0 to rank on raw ΔE_H.
    """
    required = {e.capitalize() for e in elements}
    df = load_full_dataset()
    candidates = filter_candidates(df, required, exclude_train=exclude_train)
    if candidates.empty:
        return ScreenResult(elements=sorted(required), model=model, top=top,
                             exclude_train=exclude_train, n_candidates=0,
                             dg_correction=dg_correction, rows=[], pool_dG_pred=[])

    candidates = candidates[predictable_mask(candidates, model)].copy()
    if candidates.empty:
        return ScreenResult(elements=sorted(required), model=model, top=top,
                             exclude_train=exclude_train, n_candidates=0,
                             dg_correction=dg_correction, rows=[], pool_dG_pred=[])
    ids = candidates["id"].tolist()
    if model == "ensemble":
        p_etr = predict_etr_emb(ids)
        p_sa = predict_stagea(ids)
        candidates["dE_pred"] = 0.5 * (p_etr + p_sa)
        candidates["dG_pred_etr"] = p_etr + dg_correction
        candidates["dG_pred_stagea"] = p_sa + dg_correction
    else:
        candidates["dE_pred"] = predict(ids, model)

    # error is measured in label space (model vs DFT, both without correction)
    candidates["error_vs_dft"] = candidates["dE_pred"] - candidates["delta_G_H"]
    candidates["dG_pred"] = candidates["dE_pred"] + dg_correction
    candidates["dG_dft"] = candidates["delta_G_H"] + dg_correction
    candidates["abs_dG_pred"] = candidates["dG_pred"].abs()
    top_df = candidates.sort_values("abs_dG_pred").head(top).reset_index(drop=True)

    return ScreenResult(
        elements=sorted(required), model=model, top=top,
        exclude_train=exclude_train, n_candidates=len(candidates),
        dg_correction=dg_correction, rows=top_df.to_dict(orient="records"),
        pool_dG_pred=[round(float(v), 4) for v in candidates["dG_pred"]],
    )
