"""Build the curated dataset (.traj, LMDB, SQLite) from the raw dump.

Usage:
    uv run python scripts/02_build_dataset.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from her_gnn.dataset import build_records
from her_gnn.ingest import load_raw_dump
from her_gnn.storage import write_lmdb, write_sqlite, write_traj

logger = logging.getLogger("build")

RAW_PATH = Path("data/raw/catalysis_hub_dump.json")
TRAJ_PATH = Path("data/processed/her_dataset.traj")
LMDB_PATH = Path("data/lmdb/her_dataset.lmdb")
SQLITE_PATH = Path("data/metadata.sqlite")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    nodes = load_raw_dump(RAW_PATH)
    logger.info("loaded %d raw reactions", len(nodes))

    records, stats = build_records(nodes)
    logger.info("filter/build stats: %s", dict(stats))

    write_traj(records, TRAJ_PATH)
    write_lmdb(records, LMDB_PATH)
    write_sqlite(records, SQLITE_PATH)
    logger.info("dataset built: %d structures", len(records))


if __name__ == "__main__":
    main()
