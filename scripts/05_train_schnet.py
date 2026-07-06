"""Train SchNet on the HER dataset (GPU only).

Usage:
    uv run python scripts/05_train_schnet.py --max-epochs 200 --batch-size 32 \
        --lr 1e-3 --cutoff 6.0 --run-name schnet_baseline_v1
    uv run python scripts/05_train_schnet.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

from training.evaluate import metrics_from_preds
from training.run_logger import RunLogger
from training.train import run_training

logger = logging.getLogger("schnet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-filters", type=int, default=128)
    p.add_argument("--num-interactions", type=int, default=6)
    p.add_argument("--num-gaussians", type=int, default=50)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--early-stop-monitor", default="val_r2",
                   choices=["val_loss", "val_r2", "val_mae"])
    p.add_argument("--early-stop-mode", default=None,
                   choices=["min", "max"],
                   help="default: max for val_r2, min for val_loss/val_mae")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--strategy", default="random", choices=["random", "composition"],
                   help="partição: canônica aleatória ou agrupada por composição")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="schnet_baseline")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, tiny subset, batch=4: pipeline check only")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    assert torch.cuda.is_available(), "CUDA não disponível - abortando"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    args = parse_args()
    cfg = vars(args).copy()
    if args.smoke:
        cfg["max_epochs"] = 2
        cfg["batch_size"] = 4
        cfg["patience"] = 100
        cfg["num_workers"] = 0
    cfg["ckpt_dir"] = f"logs/checkpoints/{args.run_name}"

    t0 = time.perf_counter()
    result = run_training(cfg)
    train_seconds = round(time.perf_counter() - t0, 2)
    test_metrics = metrics_from_preds(result["y_true"], result["y_pred"])
    logger.info("=== SchNet TEST METRICS ===")
    for k, v in test_metrics.items():
        logger.info("  %-5s = %.4f", k, v)

    if args.smoke:
        logger.info("smoke test OK")
        return

    run_config = {**cfg, "model": "SchNet"}
    with RunLogger(name=args.run_name, config=run_config) as run:
        run.log_metrics({
            **{f"{k}_test": v for k, v in test_metrics.items()},
            "n_params": result["n_params"],
            "vram_peak_gb": result["vram_peak_gb"],
            "best_ckpt": result["best_ckpt"],
            "elapsed_sec": train_seconds,
        })
        run.log_predictions(result["y_true"], result["y_pred"], "test",
                            sid=result["test_sids"])
        run.log_standard_figures(result["y_true"], result["y_pred"],
                                  model_label="SchNet", color="#c44e52")
        np.save(run.run_dir / "embeddings.npy", result["embeddings"])
        np.save(run.run_dir / "embeddings_y.npy", result["embeddings_y"])


if __name__ == "__main__":
    main()
