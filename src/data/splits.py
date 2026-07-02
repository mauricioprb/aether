"""Canonical dataset splits shared by every model.

Two strategies:
  - ``random``:      structure-level random split. Measures interpolation:
                     similar structures (same surface, other sites) may fall on
                     both sides.
  - ``composition``: GroupShuffleSplit by bulk composition - no composition
                     appears in more than one partition. Measures extrapolation
                     to unseen chemistries and is the leakage-robust protocol.

Each strategy persists to its own JSON so the canonical random split
(``data/splits.json``) is never clobbered. The validation subset is carved from
train by ``carve_val`` - the ONE implementation every pipeline (SchNet, MACE
fine-tune, embedding extraction) must share, so train/val/test ids are
identical across models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.20

SPLITS_PATHS = {
    "random": Path("data/splits.json"),
    "composition": Path("data/splits_composition.json"),
}
SPLITS_PATH = SPLITS_PATHS["random"]  # backward-compat alias
SQLITE_PATH = Path("data/metadata.sqlite")
VAL_FRAC = 0.1
VAL_SEED = 42


def make_splits(df: pd.DataFrame, strategy: str = "random",
                test_size: float = TEST_SIZE, seed: int = RANDOM_STATE) -> dict:
    """Split a frame with ``id`` (+ ``composition`` for the grouped strategy)."""
    ids = df["id"].astype(str).tolist()
    if strategy == "random":
        train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    elif strategy == "composition":
        groups = df["composition"].astype(str).tolist()
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(ids, groups=groups))
        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]
    else:
        raise ValueError(f"unknown split strategy: {strategy!r}")
    return {"train": train_ids, "test": test_ids, "strategy": strategy}


def carve_val(train_ids: list[str], seed: int = VAL_SEED,
              val_frac: float = VAL_FRAC) -> tuple[list[str], list[str]]:
    """Deterministically carve a validation subset out of the train ids.

    numpy RNG, matching the split used to build the published MACE embedding
    npz files. Returns (train_without_val, val).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_ids))
    n_val = int(len(train_ids) * val_frac)
    val_sel = {train_ids[k] for k in perm[:n_val]}
    return ([i for i in train_ids if i not in val_sel],
            [i for i in train_ids if i in val_sel])


def load_or_create_splits(splits_path: Path | None = None,
                          sqlite_path: Path = SQLITE_PATH,
                          strategy: str = "random") -> dict[str, list[str]]:
    """Load the persisted split for ``strategy``, creating it on first use.

    Refuses to reuse a file whose stored strategy differs from the requested
    one - mixing protocols silently would invalidate every comparison.
    """
    splits_path = Path(splits_path) if splits_path else SPLITS_PATHS[strategy]
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        stored = splits.get("strategy", "random")  # legacy files predate the key
        if stored != strategy:
            raise ValueError(
                f"{splits_path} holds a {stored!r} split but {strategy!r} was "
                "requested - use the matching splits_path"
            )
        logger.info("loaded %s splits: %d train / %d test",
                    stored, len(splits["train"]), len(splits["test"]))
        return splits

    from storage import load_features_frame

    df = load_features_frame(sqlite_path)
    splits = make_splits(df, strategy=strategy)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits))
    logger.info("created %s splits: %d train / %d test",
                strategy, len(splits["train"]), len(splits["test"]))
    return splits


def three_way_split(seed: int = VAL_SEED, val_frac: float = VAL_FRAC,
                    strategy: str = "random") -> dict[str, list[str]]:
    """Canonical train/val/test ids - the single entry point for every model."""
    splits = load_or_create_splits(strategy=strategy)
    train_ids, val_ids = carve_val(splits["train"], seed=seed, val_frac=val_frac)
    return {"train": train_ids, "val": val_ids, "test": splits["test"]}
