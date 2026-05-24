"""Build the PyG graph dataset and the canonical train/test split.

Usage:
    uv run python scripts/04_build_graphs.py [--cutoff 6.0]

Idempotent: re-running does not reprocess if the cache exists.
"""

from __future__ import annotations

import argparse
import logging

from data.dataset import HERDataset
from data.splits import load_or_create_splits

logger = logging.getLogger("graphs")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()

    dataset = HERDataset(root="data", cutoff=args.cutoff)
    logger.info("dataset ready: %d graphs (cutoff=%.1f)", len(dataset), args.cutoff)
    sample = dataset[0]
    logger.info("sample: %d nodes, %d edges", sample.num_nodes, sample.edge_index.shape[1])

    splits = load_or_create_splits()
    logger.info("splits: %d train / %d test", len(splits["train"]), len(splits["test"]))


if __name__ == "__main__":
    main()
