"""Extract MACE-MP-0 descriptors for every HER structure (GPU only).

Usage:
    uv run python scripts/07_extract_mace_features.py \
        --model medium --batch-mode sequential --output-dir data/mace_features

Saves train/val/test .npz aligned to data/splits.json (same partition as the
SchNet run). Idempotent: skips if train.npz exists unless --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from models.mace_features import (
    FEATURE_NAMES,
    delta_g_map,
    extract_features,
    three_way_split,
)
from training.run_logger import RunLogger

logger = logging.getLogger("mace")

TRAJ_PATH = Path("data/processed/her_dataset.traj")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="medium")
    p.add_argument("--batch-mode", choices=["sequential"], default="sequential")
    p.add_argument("--output-dir", type=Path, default=Path("data/mace_features"))
    p.add_argument("--cutoff-neighbors", type=float, default=2.4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def feature_stats(splits_feats: dict) -> dict:
    all_x = np.vstack([f.X for f in splits_feats.values()])
    return {
        name: {
            "mean": float(all_x[:, i].mean()),
            "std": float(all_x[:, i].std()),
            "min": float(all_x[:, i].min()),
            "max": float(all_x[:, i].max()),
        }
        for i, name in enumerate(FEATURE_NAMES)
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    assert torch.cuda.is_available(), "CUDA não disponível - abortando"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "train.npz").exists() and not args.force:
        logger.info("train.npz já existe em %s - use --force para reextrair", args.output_dir)
        return

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device="cuda", default_dtype="float32")

    splits = three_way_split()
    dgh = delta_g_map(TRAJ_PATH)
    logger.info("splits: %s", {k: len(v) for k, v in splits.items()})

    t0 = time.perf_counter()
    feats = {}
    for name, ids in splits.items():
        feats[name] = extract_features(TRAJ_PATH, ids, dgh, calc, args.cutoff_neighbors)
        feats[name].save(args.output_dir / f"{name}.npz")
        logger.info("saved %s.npz: %d structures, %d features",
                    name, len(feats[name].ids), feats[name].X.shape[1])
    elapsed = time.perf_counter() - t0
    n_total = sum(len(f.ids) for f in feats.values())
    vram = torch.cuda.max_memory_allocated() / 1e9
    logger.info("done: %d structures in %.1fs (%.1f ms/struct), VRAM peak %.2f GB",
                n_total, elapsed, 1000 * elapsed / n_total, vram)

    config = {
        "model": f"MACE-MP-0 {args.model}",
        "cutoff_neighbors": args.cutoff_neighbors,
        "feature_names": FEATURE_NAMES,
        "n_train": len(feats["train"].ids),
        "n_val": len(feats["val"].ids),
        "n_test": len(feats["test"].ids),
    }
    with RunLogger(name="mace_extraction", config=config) as run:
        run.log_metrics({
            "n_structures": n_total,
            "n_features": len(FEATURE_NAMES),
            "avg_infer_ms": round(1000 * elapsed / n_total, 2),
            "vram_peak_gb": round(vram, 3),
            "elapsed_sec": round(elapsed, 2),
        })
        (run.run_dir / "feature_stats.json").write_text(
            json.dumps(feature_stats(feats), indent=2))
    logger.info("RunLogger entry + feature_stats.json written")


if __name__ == "__main__":
    main()
