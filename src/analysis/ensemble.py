"""Prediction ensembling across logged runs.

Combines per-run ``predictions.parquet`` files (aligned by ``sid``) into
ensemble predictions. Members are averaged with uniform weights - nothing is
fitted on the test set. An optional oracle sweep quantifies the upper bound of
pairwise linear blending; its weights ARE chosen on test, so it is a
diagnostic, never a reportable result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from training.evaluate import metrics_from_preds

logger = logging.getLogger(__name__)

RUNS_DIR = Path("results/runs")
Y_TRUE_ATOL = 1e-6


def run_name(run_dir: Path) -> str:
    """`{YYYYMMDD}_{HHMMSS}_{name}` -> `name`."""
    return "_".join(run_dir.name.split("_")[2:])


def latest_runs_by_name(names: list[str], runs_dir: Path = RUNS_DIR) -> dict[str, Path]:
    """Latest run dir for each exact run name. Raises if any name is missing."""
    found: dict[str, Path] = {}
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        name = run_name(d)
        if name in names:
            found[name] = d  # sorted -> the last (latest timestamp) wins
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(f"no run found in {runs_dir} for: {missing}")
    return found


def load_split_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    """Load ``predictions.parquet`` for one split, indexed by sid."""
    df = pd.read_parquet(run_dir / "predictions.parquet")
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"{run_dir} has no '{split}' predictions logged")
    if (df["sid"] == "").any():
        raise ValueError(f"{run_dir} has {split} predictions without sid - cannot align")
    return df.set_index("sid")[["y_true", "y_pred"]].sort_index()


def load_test_predictions(run_dir: Path) -> pd.DataFrame:
    """Load ``predictions.parquet`` restricted to the test split, indexed by sid."""
    return load_split_predictions(run_dir, "test")


def align_members(member_dirs: dict[str, Path], split: str = "test",
                  ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align member predictions by sid, for the given split.

    Returns (sids, y_true, preds) with ``preds`` of shape (n_members, n_samples).
    Raises if members disagree on the sid set or on y_true (split drift guard).
    """
    frames = {name: load_split_predictions(d, split) for name, d in member_dirs.items()}
    names = list(frames)
    base = frames[names[0]]
    sids = base.index.tolist()
    y_true = base["y_true"].to_numpy()

    preds = np.empty((len(names), len(sids)))
    for k, name in enumerate(names):
        df = frames[name]
        if df.index.tolist() != sids:
            raise ValueError(f"member {name}: test sid set differs from {names[0]}")
        if not np.allclose(df["y_true"].to_numpy(), y_true, atol=Y_TRUE_ATOL):
            raise ValueError(f"member {name}: y_true differs from {names[0]} - "
                             "runs used different targets/splits")
        preds[k] = df["y_pred"].to_numpy()
    return sids, y_true, preds


@dataclass
class EnsembleResult:
    name: str
    members: list[str]
    sids: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]


def uniform_ensemble(name: str, member_dirs: dict[str, Path]) -> EnsembleResult:
    """Mean of member predictions (uniform weights, nothing fitted)."""
    sids, y_true, preds = align_members(member_dirs)
    y_pred = preds.mean(axis=0)
    return EnsembleResult(
        name=name,
        members=list(member_dirs),
        sids=sids,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics_from_preds(y_true, y_pred),
    )


def oracle_pair_blend(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                      step: float = 0.05) -> tuple[float, dict[str, float]]:
    """Best w for ``w*a + (1-w)*b`` chosen ON TEST - diagnostic upper bound only."""
    best_w, best_m = 0.0, None
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        m = metrics_from_preds(y_true, w * pred_a + (1 - w) * pred_b)
        if best_m is None or m["r2"] > best_m["r2"]:
            best_w, best_m = float(w), m
    return best_w, best_m


def val_fitted_blend(name: str, member_dirs: dict[str, Path]) -> tuple[EnsembleResult, np.ndarray]:
    """Non-negative blend weights fitted on VALIDATION, applied to test.

    Legitimate (test never sees weight selection), unlike ``oracle_pair_blend``.
    Requires every member to have logged 'val' predictions. Returns the test
    result plus the fitted weight vector (member order = dict order).
    """
    from scipy.optimize import nnls

    _, y_val, val_preds = align_members(member_dirs, split="val")
    weights, _ = nnls(val_preds.T, y_val)
    if weights.sum() == 0:
        weights = np.ones(len(member_dirs))
    weights = weights / weights.sum()

    sids, y_true, test_preds = align_members(member_dirs, split="test")
    y_pred = weights @ test_preds
    result = EnsembleResult(
        name=name, members=list(member_dirs), sids=sids, y_true=y_true,
        y_pred=y_pred, metrics=metrics_from_preds(y_true, y_pred),
    )
    return result, weights
