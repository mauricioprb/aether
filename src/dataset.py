from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from tqdm import tqdm

from features import compute_features
from filters import classify_site, is_clean_her, parse_coverage, passes_filters
from ingest import reaction_to_atoms

logger = logging.getLogger(__name__)


def build_records(
    nodes: list[dict[str, Any]], source: str = "catalysis_hub"
) -> tuple[list[dict[str, Any]], Counter]:
    """Build curated records from raw reaction nodes. Returns ``(records, stats)``."""
    records: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    for node in tqdm(nodes, desc="building records"):
        stats["total"] += 1
        if node["id"] in seen:
            stats["duplicate"] += 1
            continue
        seen.add(node["id"])
        if not is_clean_her(node):
            stats["not_clean_her"] += 1
            continue
        atoms = reaction_to_atoms(node)
        if atoms is None:
            stats["no_structure"] += 1
            continue

        coverage = parse_coverage(node)
        delta_g = node["reactionEnergy"]
        ok, reason = passes_filters(atoms, delta_g, coverage)
        if not ok:
            stats[f"reject:{reason}"] += 1
            continue

        site, _ = classify_site(atoms)
        features = compute_features(atoms)
        if features is None:
            stats["no_features"] += 1
            continue

        records.append({
            "id": node["id"],
            "atoms": atoms,
            "delta_G_H": delta_g,
            "coverage": coverage,
            "site_type": site,
            "composition": node["chemicalComposition"],
            "chemical_formula": atoms.get_chemical_formula(),
            "facet": node["facet"],
            "n_atoms": len(atoms),
            "source": source,
            "features": features,
        })
        stats["kept"] += 1
    logger.info("kept %d / %d records", stats["kept"], stats["total"])
    return records, stats
