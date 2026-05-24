from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from baseline import RANDOM_STATE, TEST_SIZE
from storage import load_features_frame

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("data/splits.json")
SQLITE_PATH = Path("data/metadata.sqlite")


def load_or_create_splits(splits_path: Path = SPLITS_PATH,
                          sqlite_path: Path = SQLITE_PATH) -> dict[str, list[str]]:
    splits_path = Path(splits_path)
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        logger.info("loaded splits: %d train / %d test", len(splits["train"]), len(splits["test"]))
        return splits

    df = load_features_frame(sqlite_path)
    ids = df["id"].tolist()
    train_ids, test_ids = train_test_split(ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    splits = {"train": train_ids, "test": test_ids}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits))
    logger.info("created splits: %d train / %d test", len(train_ids), len(test_ids))
    return splits
