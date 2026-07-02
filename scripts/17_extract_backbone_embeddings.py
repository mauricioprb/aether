"""Extract per-structure embeddings from any registered pretrained backbone.

Generalizes 09_extract_mace_embeddings to the ``BACKBONES`` registry, so the
representation comparison (MACE-MP-0 vs OC20-pretrained vs ...) is one command
per backbone followed by the standard ETR sweep:

    uv run python scripts/17_extract_backbone_embeddings.py --backbone eqv2_oc20
    uv run python scripts/08b_feature_reduction_sweep.py --feature-set emb \\
        --features-dir data/backbone_features/eqv2_oc20

Output: data/backbone_features/{backbone}/{train,val,test}_emb.npz, same npz
contract as the MACE pipeline (ids, X, y, feature_names).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ase.io import Trajectory
from tqdm import tqdm

from data.splits import three_way_split
from models.backbones import BACKBONES, pool_descriptors
from models.mace_features import MACEFeatures, delta_g_map, embedding_names
from training.run_logger import RunLogger

logger = logging.getLogger("backbone-emb")

TRAJ_PATH = Path("data/processed/her_dataset.traj")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: data/backbone_features/{backbone}")
    p.add_argument("--cutoff-neighbors", type=float, default=2.4)
    p.add_argument("--strategy", choices=["random", "composition"], default="random",
                   help="which canonical split to align the npz files to")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = args.output_dir or Path("data/backbone_features") / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train_emb.npz").exists() and not args.force:
        logger.info("%s/train_emb.npz já existe - use --force", out_dir)
        return

    backbone = BACKBONES[args.backbone]()
    splits = three_way_split(strategy=args.strategy)
    dgh = delta_g_map(TRAJ_PATH)
    frames = {a.info["id"]: a for a in Trajectory(str(TRAJ_PATH))}
    logger.info("backbone=%s splits=%s", backbone.name,
                {k: len(v) for k, v in splits.items()})

    t0 = time.perf_counter()
    n_total, dim = 0, None
    import numpy as np

    for split_name, ids in splits.items():
        rows, kept, ys = [], [], []
        for sid in tqdm(ids, desc=f"{backbone.name} {split_name}"):
            atoms = frames.get(sid)
            if atoms is None:
                continue
            desc = backbone.descriptors(atoms)
            vec = pool_descriptors(desc, atoms, args.cutoff_neighbors)
            dim = dim or len(vec) // 2
            rows.append(vec)
            kept.append(sid)
            ys.append(dgh[sid])
        feats = MACEFeatures(ids=np.array(kept, dtype=str),
                             X=np.vstack(rows).astype(np.float32),
                             y=np.array(ys, dtype=np.float32),
                             feature_names=embedding_names(dim))
        feats.save(out_dir / f"{split_name}_emb.npz")
        n_total += len(kept)
        logger.info("saved %s: %d structures, %d features",
                    out_dir / f"{split_name}_emb.npz", len(kept), feats.X.shape[1])

    elapsed = time.perf_counter() - t0
    config = {"backbone": backbone.name, "strategy": args.strategy,
              "pooling": "[emb(H), mean(emb(central))]",
              "cutoff_neighbors": args.cutoff_neighbors, "dim": 2 * (dim or 0)}
    with RunLogger(name=f"emb_{backbone.name}", config=config) as run:
        run.log_metrics({"n_structures": n_total, "n_features": 2 * (dim or 0),
                         "avg_infer_ms": round(1000 * elapsed / max(n_total, 1), 2),
                         "elapsed_sec": round(elapsed, 2)})
    logger.info("done: %d structures in %.1fs", n_total, elapsed)


if __name__ == "__main__":
    main()
