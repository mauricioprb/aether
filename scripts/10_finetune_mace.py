"""Fine-tune MACE-MP-0 for ΔG_H prediction (GPU only).

Usage:
    # Stage A: frozen backbone + trainable MLP head (fast, ~2 GB VRAM)
    uv run python scripts/10_finetune_mace.py --freeze-backbone --run-name mace_ft_stageA

    # Stage B: unfreeze last interaction layer
    uv run python scripts/10_finetune_mace.py --freeze-backbone --unfreeze-last-n 1 --run-name mace_ft_stageB

    # Stage C: full fine-tune (low lr, ~6 GB VRAM)
    uv run python scripts/10_finetune_mace.py --no-freeze-backbone --lr 1e-5 --run-name mace_ft_stageC

    # Smoke test (2 epochs, tiny subset)
    uv run python scripts/10_finetune_mace.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data.mace_dataset import MACEDataset, three_way_split_mace
from models.mace_finetune import LitMACEFineTune
from training.evaluate import metrics_from_preds
from training.run_logger import RunLogger

logger = logging.getLogger("mace-finetune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Model
    p.add_argument("--mace-model", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    # Freeze strategy
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                   help="Freeze MACE backbone (default: True)")
    p.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
                   help="Unfreeze entire MACE backbone")
    p.add_argument("--unfreeze-last-n", type=int, default=0,
                   help="Unfreeze last N interaction layers (requires --freeze-backbone)")
    # Training
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--lr-patience", type=int, default=20)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=60,
                   help="Early stopping patience (epochs)")
    p.add_argument("--early-stop-monitor", default="val_r2",
                   choices=["val_loss", "val_r2", "val_mae"])
    p.add_argument("--early-stop-mode", default=None,
                   choices=["min", "max"],
                   help="default: max for val_r2, min for val_loss/val_mae")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--run-name", default="mace_finetune")
    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, tiny subset: pipeline check only")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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

    # --- Data ---
    splits = three_way_split_mace(seed=args.seed, val_frac=args.val_frac)
    logger.info("split: train=%d val=%d test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # We need z_table/r_max from the model first; create a temporary calc to get them
    from mace.calculators import mace_mp
    tmp = mace_mp(model=args.mace_model, device="cuda", default_dtype="float32")
    z_table = tmp.z_table
    r_max = float(tmp.models[0].r_max.item())
    del tmp
    torch.cuda.empty_cache()

    common = dict(z_table=z_table, r_max=r_max)
    if args.smoke:
        all_ids = splits["train"][:40] + splits["val"][:12] + splits["test"][:12]
        train_ids = all_ids[:40]
        val_ids = all_ids[40:52]
        test_ids = all_ids[52:64]
    else:
        train_ids = splits["train"]
        val_ids = splits["val"]
        test_ids = splits["test"]

    train_ds = MACEDataset(ids=train_ids, **common)
    val_ds = MACEDataset(ids=val_ids, **common)
    test_ds = MACEDataset(ids=test_ids, **common)

    dl_common = dict(num_workers=cfg["num_workers"], pin_memory=True,
                     persistent_workers=cfg["num_workers"] > 0,
                     multiprocessing_context="spawn" if cfg["num_workers"] > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **dl_common)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], **dl_common)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], **dl_common)

    # --- Model ---
    model = LitMACEFineTune(
        mace_model=args.mace_model,
        mace_device="cuda",
        default_dtype="float32",
        hidden_dim=args.hidden_dim,
        head_layers=args.head_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
    )
    y_train = torch.tensor([d.y.item() for d in train_ds])
    model.set_target_stats(y_train.mean().item(), y_train.std().item())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("params: %d trainable / %d total", n_params, n_total)

    # --- Training ---
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    seed_everything(args.seed, workers=True)

    monitor = args.early_stop_monitor
    mode = args.early_stop_mode or ("max" if monitor == "val_r2" else "min")
    ckpt = ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=3, save_last=True,
                           dirpath=cfg["ckpt_dir"])
    early_stop = EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
    tb_logger = TensorBoardLogger(save_dir="logs", name=args.run_name)

    t0 = time.perf_counter()
    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu", devices=1, precision="16-mixed",
        gradient_clip_val=10.0,
        callbacks=[ckpt, early_stop],
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=not args.smoke,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")
    train_seconds = round(time.perf_counter() - t0, 2)

    # --- Metrics ---
    y_true = np.array(model.test_true)
    y_pred = np.array(model.test_pred)
    test_metrics = metrics_from_preds(y_true, y_pred)
    vram = torch.cuda.max_memory_allocated() / 1e9

    logger.info("=== MACE fine-tune TEST METRICS ===")
    for k, v in test_metrics.items():
        logger.info("  %-5s = %.4f", k, v)
    logger.info("VRAM peak: %.2f GB | trainable params: %d | time: %.1f s",
                vram, n_params, train_seconds)

    if args.smoke:
        logger.info("smoke test OK")
        return

    # --- Log results ---
    run_config = {**cfg, "model": f"MACE-MP-0 {args.mace_model} fine-tune",
                  "n_trainable": n_params, "n_total": n_total,
                  "freeze_backbone": args.freeze_backbone,
                  "unfreeze_last_n": args.unfreeze_last_n}
    with RunLogger(name=args.run_name, config=run_config) as run:
        run.log_metrics({
            **{f"{k}_test": v for k, v in test_metrics.items()},
            "n_params": n_params,
            "vram_peak_gb": round(vram, 3),
            "best_ckpt": ckpt.best_model_path,
            "elapsed_sec": train_seconds,
        })
        run.log_predictions(y_true, y_pred, "test", sid=test_ds.sids)

        model.eval().cuda()
        val_pred, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.cuda()
                val_pred.extend(model(batch).cpu().tolist())
                val_true.extend(batch.y.view(-1).cpu().tolist())
        run.log_predictions(np.array(val_true), np.array(val_pred), "val",
                            sid=val_ds.sids)

        run.log_standard_figures(y_true, y_pred,
                                  model_label=f"MACE {args.run_name}",
                                  color="#4c72b0")

        model.eval().cuda()
        emb_list, emb_y = [], []
        for batch in test_loader:
            batch = batch.cuda()
            emb_list.append(model.embed(batch).cpu().numpy())
            emb_y.extend(batch.y.view(-1).cpu().tolist())
        np.save(run.run_dir / "embeddings.npy", np.vstack(emb_list))
        np.save(run.run_dir / "embeddings_y.npy", np.array(emb_y))


if __name__ == "__main__":
    main()
