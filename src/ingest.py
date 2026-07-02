"""Catalysis Hub GraphQL ingest.

Fetches HER adsorption reactions (0.5 H2 + * -> H*) and reconstructs the final
slab-with-adsorbed-H structures as ``ase.Atoms``. The structure for each reaction
lives in its ``reactionSystems``: ``star`` is the bare surface, ``H2gas`` the gas
reference, and the H-bearing product (e.g. ``Hstar``) is the adsorbed slab we want.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from ase import Atoms
from ase.db import connect
from tqdm import tqdm

logger = logging.getLogger(__name__)

GRAPHQL_ENDPOINT = "https://api.catalysis-hub.org/graphql"

REACTION_FIELDS = """
id
Equation
chemicalComposition
surfaceComposition
reactionEnergy
activationEnergy
facet
sites
coverages
reactants
products
pubId
dftFunctional
"""

_SYSTEMS_BLOCK = """
        reactionSystems {
          name
          systems {
            energy
            InputFile(format: "json")
          }
        }
"""

_NODE_FIELDS = REACTION_FIELDS + _SYSTEMS_BLOCK

_REACTIONS_QUERY = (
    """
query Reactions($first: Int!, $products: String!, $after: String) {
  reactions(first: $first, products: $products, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
"""
    + _NODE_FIELDS
    + """
      }
    }
  }
}
"""
)

_METADATA_QUERY = (
    """
query Metadata($first: Int!, $products: String!, $after: String) {
  reactions(first: $first, products: $products, after: $after, order: "id") {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
"""
    + REACTION_FIELDS
    + """
      }
    }
  }
}
"""
)


def reaction_pk(reaction_id: str) -> int:
    """Numeric primary key from a base64 reaction id ('Reaction:455114' -> 455114)."""
    return int(base64.b64decode(reaction_id).decode().split(":")[1])


def graphql_query(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
    retries: int = 4,
    backoff: float = 2.0,
) -> dict[str, Any]:
    """POST a GraphQL query, retrying transient failures with exponential backoff."""
    payload = {"query": query, "variables": variables or {}}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(GRAPHQL_ENDPOINT, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                raise RuntimeError(f"GraphQL errors: {body['errors']}")
            return body["data"]
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            wait = backoff ** attempt
            logger.warning("query attempt %d/%d failed: %s; retry in %.1fs",
                           attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GraphQL query failed after {retries} attempts") from last_exc


def fetch_reactions(
    first: int = 10,
    products: str = "H",
    after: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch one page of reactions. Returns ``(nodes, page_info)``."""
    data = graphql_query(_REACTIONS_QUERY,
                         {"first": first, "products": products, "after": after})
    conn = data["reactions"]
    nodes = [edge["node"] for edge in conn["edges"]]
    logger.info("fetched %d reactions (hasNextPage=%s)",
                len(nodes), conn["pageInfo"]["hasNextPage"])
    return nodes, conn["pageInfo"]


def fetch_all_reactions(
    products: str = "H",
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Paginate the full reactions connection until exhausted (or ``max_records``)."""
    nodes: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        page, info = fetch_reactions(first=page_size, products=products, after=after)
        nodes.extend(page)
        if max_records is not None and len(nodes) >= max_records:
            return nodes[:max_records]
        if not info["hasNextPage"]:
            return nodes
        after = info["endCursor"]


def fetch_all_metadata(products: str = "H", page_size: int = 1000) -> list[dict[str, Any]]:
    """Paginate lightweight reaction metadata (no structures) until exhausted."""
    by_id: dict[str, dict[str, Any]] = {}
    after: str | None = None
    while True:
        data = graphql_query(_METADATA_QUERY,
                            {"first": page_size, "products": products, "after": after})
        conn = data["reactions"]
        for edge in conn["edges"]:
            by_id[edge["node"]["id"]] = edge["node"]
        logger.info("metadata: %d unique reactions", len(by_id))
        if not conn["pageInfo"]["hasNextPage"]:
            return list(by_id.values())
        after = conn["pageInfo"]["endCursor"]


def fetch_reactions_by_ids(
    reaction_ids: list[str], batch_size: int = 40
) -> list[dict[str, Any]]:
    """Fetch full reactions (with structures) for specific ids via aliased batches."""
    nodes: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(reaction_ids), batch_size), desc="fetch structures"):
        chunk = reaction_ids[start : start + batch_size]
        aliases = "\n".join(
            f'r{reaction_pk(rid)}: reactions(id: {reaction_pk(rid)}) '
            f"{{ edges {{ node {{ {_NODE_FIELDS} }} }} }}"
            for rid in chunk
        )
        data = graphql_query("{\n" + aliases + "\n}")
        for value in data.values():
            edges = value.get("edges", [])
            if edges:
                nodes.append(edges[0]["node"])
    logger.info("fetched %d reactions by id", len(nodes))
    return nodes


def system_to_atoms(input_file_json: str) -> Atoms:
    """Convert a Catalysis Hub ``InputFile(format:"json")`` (ASE db dump) to ``Atoms``."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write(input_file_json)
        path = fh.name
    try:
        return connect(path).get_atoms(id=1)
    finally:
        Path(path).unlink(missing_ok=True)


def pick_adsorbate_system(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return the reaction system holding the adsorbed slab (H-bearing product).

    Prefers a product named ``Hstar``; falls back to any product name containing
    ``star`` other than the bare ``star`` surface.
    """
    systems = {rs["name"]: rs["systems"] for rs in node["reactionSystems"]}
    if "Hstar" in systems:
        return systems["Hstar"]
    for name, sysd in systems.items():
        if name != "star" and name.endswith("star"):
            return sysd
    return None


def reaction_to_atoms(node: dict[str, Any]) -> Atoms | None:
    """Reconstruct the adsorbed-slab ``Atoms`` for a reaction, attaching ``info`` metadata."""
    sysd = pick_adsorbate_system(node)
    if sysd is None or not sysd.get("InputFile"):
        logger.debug("no adsorbate system for reaction %s", node.get("id"))
        return None
    atoms = system_to_atoms(sysd["InputFile"])
    atoms.info.update({
        "id": node["id"],
        # ATENCAO: reactionEnergy do Catalysis Hub e a energia ELETRONICA
        # dE_H (cathub deriva de get_potential_energy(); Mamun 2019, Eq. 1).
        # A chave mantem o nome legado "delta_G_H" por compatibilidade com os
        # artefatos ja gerados; dG_H = dE_H + 0.24 eV entra so no screening.
        "delta_G_H": node["reactionEnergy"],
        "source": "catalysis_hub",
        "composition": node["chemicalComposition"],
        "facet": node["facet"],
        "pub_id": node.get("pubId", ""),
        "dft_functional": node.get("dftFunctional", ""),
    })
    return atoms


def save_raw_dump(nodes: list[dict[str, Any]], path: str | Path) -> Path:
    """Persist raw reaction nodes as JSON for reproducibility / offline dev."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nodes, indent=2))
    logger.info("wrote %d reactions to %s", len(nodes), path)
    return path


def load_raw_dump(path: str | Path) -> list[dict[str, Any]]:
    """Load a previously saved raw dump."""
    return json.loads(Path(path).read_text())
