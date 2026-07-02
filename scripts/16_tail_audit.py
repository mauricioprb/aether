"""Tail audit: the worst test-set errors, cross-referenced across models.

The MSE (and hence R²/RMSE) is dominated by a handful of catastrophic errors.
This script ranks test structures by |error| across the selected runs and
flags the recurring offenders. A structure that EVERY model misses badly is a
label/curation suspect (H desorbed, site migration, duplicate with divergent
labels); a structure only one model misses is a model problem.

Works from ``results/runs/*/predictions.parquet`` alone. When
``data/metadata.sqlite`` and ``data/processed/her_dataset.traj`` are present,
rows are enriched with formula/facet/site and H-surface geometry checks.

Usage:
    uv run python scripts/16_tail_audit.py                # top 30, all groups
    uv run python scripts/16_tail_audit.py --top 50
    uv run python scripts/16_tail_audit.py --threshold 0.5  # |err| > 0.5 eV
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.ensemble import latest_runs_by_name, load_test_predictions

logger = logging.getLogger("tail-audit")

SEEDS = [42, 1, 2, 3, 4]
MODEL_GROUPS = {
    "stagea": [f"mace_ft_stageA_v2_seed{s}" for s in SEEDS],
    "schnet": [f"schnet_v2_seed{s}" for s in SEEDS],
    "etr_emb": ["etr_emb_all"],
    "etr_baseline": ["etr_baseline"],
}
SQLITE_PATH = Path("data/metadata.sqlite")
TRAJ_PATH = Path("data/processed/her_dataset.traj")
OUT_CSV = Path("results/tail_audit.csv")


def group_abs_error(names: list[str]) -> pd.Series:
    """Mean |error| per sid over the runs in the group (seed-averaged)."""
    dirs = latest_runs_by_name(names)
    per_run = []
    for d in dirs.values():
        df = load_test_predictions(d)
        per_run.append((df["y_pred"] - df["y_true"]).abs())
    return pd.concat(per_run, axis=1).mean(axis=1)


def load_metadata() -> pd.DataFrame | None:
    if not SQLITE_PATH.exists() or SQLITE_PATH.stat().st_size == 0:
        logger.warning("no %s - skipping formula/facet enrichment", SQLITE_PATH)
        return None
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        cols = "id, chemical_formula, facet, site_type, coverage, delta_G_H"
        have = {r[1] for r in conn.execute("PRAGMA table_info(structures)")}
        # pub_id/dft_functional só existem em bancos reconstruídos após a
        # descoberta da heterogeneidade de fontes (Mamun/BEEF-vdW vs
        # Yohannes/PBE); erros concentrados em uma fonte apontam
        # incompatibilidade de funcional, não falha do modelo.
        if "pub_id" in have:
            cols += ", pub_id, dft_functional"
        return pd.read_sql_query(
            f"SELECT {cols} FROM structures", conn,
        ).set_index("id")
    finally:
        conn.close()


def geometry_checks(sids: list[str]) -> pd.DataFrame | None:
    """H-surface distance + H coordination for the flagged structures."""
    if not TRAJ_PATH.exists():
        logger.warning("no %s - skipping geometry checks", TRAJ_PATH)
        return None
    from ase.io import Trajectory

    from geometry import central_indices, h_surface_min_distance

    wanted = set(sids)
    rows = []
    for atoms in Trajectory(str(TRAJ_PATH)):
        sid = str(atoms.info.get("id"))
        if sid not in wanted:
            continue
        rows.append({
            "sid": sid,
            "h_surf_dist": h_surface_min_distance(atoms),
            "h_coordination": len(central_indices(atoms)),
            "n_atoms": len(atoms),
        })
    return pd.DataFrame(rows).set_index("sid") if rows else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=30, help="worst-N structures to list")
    ap.add_argument("--threshold", type=float, default=None,
                    help="instead of top-N, list all with mean |err| above this (eV)")
    args = ap.parse_args()

    errors = {}
    for group, names in MODEL_GROUPS.items():
        try:
            errors[group] = group_abs_error(names)
        except FileNotFoundError as exc:
            logger.warning("group %s skipped: %s", group, exc)
    if not errors:
        raise SystemExit("no runs found under results/runs/")

    tab = pd.DataFrame(errors)
    tab["mean_abs_err"] = tab.mean(axis=1)
    tab["n_models_gt_0.5eV"] = (tab[list(errors)] > 0.5).sum(axis=1)
    # y_true from any group's first run (aligned by construction)
    first_dir = next(iter(latest_runs_by_name(MODEL_GROUPS["etr_emb"]).values()))
    tab["y_true"] = load_test_predictions(first_dir)["y_true"]

    tab = tab.sort_values("mean_abs_err", ascending=False)
    flagged = tab[tab["mean_abs_err"] > args.threshold] if args.threshold else tab.head(args.top)

    meta = load_metadata()
    if meta is not None:
        flagged = flagged.join(meta, how="left")
        if "pub_id" in meta.columns:
            per_src = (tab.join(meta[["pub_id", "dft_functional"]], how="left")
                       .groupby(["pub_id", "dft_functional"])["mean_abs_err"]
                       .agg(n="count", mae_medio="mean", mae_mediano="median"))
            print("\nErro (média entre modelos) por fonte do dataset:")
            print(per_src.to_string(float_format="%.3f"), "\n")
    geo = geometry_checks(flagged.index.tolist())
    if geo is not None:
        flagged = flagged.join(geo, how="left")

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    n_all = len(tab)
    consensual = int((tab["n_models_gt_0.5eV"] >= len(errors)).sum())
    print(f"\ntest set: {n_all} structures | groups: {list(errors)}")
    print(f"structures missed >0.5 eV by ALL groups (curation suspects): {consensual}")
    mse_all = float((tab["mean_abs_err"] ** 2).mean())
    mse_tail = float((flagged["mean_abs_err"] ** 2).sum()) / n_all
    print(f"tail contribution: the {len(flagged)} flagged structures hold "
          f"{100 * mse_tail / mse_all:.1f}% of the (seed-averaged) MSE\n")
    print(flagged.to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(OUT_CSV)
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
