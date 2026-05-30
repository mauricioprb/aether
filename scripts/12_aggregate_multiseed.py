"""Aggregate multi-seed runs into mean ± std markdown tables.

Reads results/summary.json, groups runs by name prefix, computes statistics,
and writes results/multiseed_summary.md (per-seed + grouped tables).

Usage:
    uv run python scripts/12_aggregate_multiseed.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from analysis.multiseed import (
    DEFAULT_GROUPS,
    aggregate_groups,
    markdown_table,
    per_seed_table,
)

logger = logging.getLogger("aggregate")

RESULTS = Path("results")
OUT_PATH = RESULTS / "multiseed_summary.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = aggregate_groups(DEFAULT_GROUPS)
    if not results:
        logger.error("no multi-seed groups found in results/summary.json")
        return

    sections = ["# Multi-seed summary\n",
                "## Grouped (mean ± std)\n",
                markdown_table(results),
                "\n## Per-seed (sanity inspection)\n",
                per_seed_table(results)]
    OUT_PATH.write_text("\n".join(sections))
    print("\n".join(sections))
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
