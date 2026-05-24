"""Extract MACE-MP-0 invariant node embeddings per HER structure (GPU only).

Usage:
    uv run python scripts/09_extract_mace_embeddings.py

Pooling: [emb(H), mean(emb(neighbors<2.4A))] from invariant (L=0) descriptors ->
512 features. Saves data/mace_features/{train,val,test}_emb.npz aligned to the
canonical split. Idempotent (--force to redo). ~10 min on the RTX 5060 Ti.
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
    delta_g_map,
    extract_embeddings,
    three_way_split,
)
from training.run_logger import RunLogger

logger = logging.getLogger("mace-emb")

TRAJ_PATH = Path("data/processed/her_dataset.traj")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    assert torch.cuda.is_available(), "CUDA não disponível - abortando"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="medium")
    p.add_argument("--output-dir", type=Path, default=Path("data/mace_features"))
    p.add_argument("--cutoff-neighbors", type=float, default=2.4)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "train_emb.npz").exists() and not args.force:
        logger.info("train_emb.npz já existe - use --force para reextrair")
        return

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device="cuda", default_dtype="float32")

    splits = three_way_split()
    dgh = delta_g_map(TRAJ_PATH)
    logger.info("splits: %s", {k: len(v) for k, v in splits.items()})

    t0 = time.perf_counter()
    feats = {}
    for name, ids in splits.items():
        feats[name] = extract_embeddings(TRAJ_PATH, ids, dgh, calc, args.cutoff_neighbors)
        feats[name].save(args.output_dir / f"{name}_emb.npz")
        logger.info("saved %s_emb.npz: %d structures, %d features",
                    name, len(feats[name].ids), feats[name].X.shape[1])
    elapsed = time.perf_counter() - t0
    n_total = sum(len(f.ids) for f in feats.values())
    vram = torch.cuda.max_memory_allocated() / 1e9
    logger.info("done: %d structures, %d-dim, %.1fs (%.1f ms/struct), VRAM %.2f GB",
                n_total, feats["train"].X.shape[1], elapsed, 1000 * elapsed / n_total, vram)

    config = {"model": f"MACE-MP-0 {args.model}", "kind": "invariant_node_embeddings",
              "pooling": "[emb(H), mean(emb(neighbors))]", "dim": feats["train"].X.shape[1],
              "cutoff_neighbors": args.cutoff_neighbors,
              "n_train": len(feats["train"].ids), "n_val": len(feats["val"].ids),
              "n_test": len(feats["test"].ids)}
    with RunLogger(name="mace_embeddings", config=config) as run:
        run.log_metrics({"n_structures": n_total, "n_features": feats["train"].X.shape[1],
                         "avg_infer_ms": round(1000 * elapsed / n_total, 2),
                         "vram_peak_gb": round(vram, 3), "elapsed_sec": round(elapsed, 2)})
        all_x = np.vstack([f.X for f in feats.values()])
        (run.run_dir / "embedding_stats.json").write_text(json.dumps(
            {"dim": int(all_x.shape[1]), "global_mean": float(all_x.mean()),
             "global_std": float(all_x.std())}, indent=2))
    logger.info("RunLogger entry written")


if __name__ == "__main__":
    main()
