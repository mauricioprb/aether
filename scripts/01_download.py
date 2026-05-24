"""Download HER structures from Catalysis Hub via a two-pass strategy.

products="H" returns ~60k reactions, most of them co-adsorption (H2S, NH3, ...).
Pass 1 pulls lightweight metadata for all of them; pass 2 fetches structures only
for the clean HER reactions (0.5 H2 + * -> H*) that survive the metadata filters.

Usage:
    uv run python scripts/01_download.py [--batch-size 40]

Outputs:
    data/raw/catalysis_hub_meta.json   all reaction metadata (pass 1 cache)
    data/raw/catalysis_hub_dump.json   clean-HER reactions with structures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from her_gnn.filters import passes_metadata_filters
from her_gnn.ingest import fetch_all_metadata, fetch_reactions_by_ids, save_raw_dump

logger = logging.getLogger("download")

META_PATH = Path("data/raw/catalysis_hub_meta.json")
DUMP_PATH = Path("data/raw/catalysis_hub_dump.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    parser.add_argument("--out", type=Path, default=DUMP_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.meta.exists():
        meta = json.loads(args.meta.read_text())
        logger.info("loaded cached metadata: %d reactions", len(meta))
    else:
        meta = fetch_all_metadata(products="H")
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(meta))
        logger.info("saved metadata: %d reactions", len(meta))

    candidates = list({n["id"] for n in meta if passes_metadata_filters(n)})
    logger.info("clean-HER candidates after metadata filters: %d", len(candidates))

    nodes = fetch_reactions_by_ids(candidates, batch_size=args.batch_size)
    save_raw_dump(nodes, args.out)
    logger.info("done: %d reactions with structures -> %s", len(nodes), args.out)


if __name__ == "__main__":
    main()
