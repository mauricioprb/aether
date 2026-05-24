from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from data.dataset import HERDataset
from data.splits import load_or_create_splits
from models.schnet import LitSchNet

logger = logging.getLogger(__name__)


def _id_to_index(dataset: HERDataset) -> dict[str, int]:
    return {d.sid: i for i, d in enumerate(dataset)}


def make_subsets(dataset: HERDataset, val_frac: float = 0.1, seed: int = 42):
    splits = load_or_create_splits()
    idx = _id_to_index(dataset)
    train_ids = [i for i in splits["train"] if i in idx]
    test_ids = [i for i in splits["test"] if i in idx]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_ids))
    n_val = int(len(train_ids) * val_frac)
    val_sel = {train_ids[k] for k in perm[:n_val]}

    train_idx = [idx[i] for i in train_ids if i not in val_sel]
    val_idx = [idx[i] for i in train_ids if i in val_sel]
    test_idx = [idx[i] for i in test_ids]
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int = 4):
    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_ds, batch_size=batch_size, **common),
        DataLoader(test_ds, batch_size=batch_size, **common),
    )


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    pl.seed_everything(cfg["seed"], workers=True)
    dataset = HERDataset(root="data", cutoff=cfg["cutoff"])

    if cfg.get("smoke"):
        dataset = dataset[:64]
        train_ds, val_ds, test_ds = dataset[:40], dataset[40:52], dataset[52:]
    else:
        train_ds, val_ds, test_ds = make_subsets(dataset, val_frac=cfg["val_frac"], seed=cfg["seed"])
    logger.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, cfg["batch_size"], cfg["num_workers"])

    model = LitSchNet(
        hidden_channels=cfg["hidden_channels"], num_filters=cfg["num_filters"],
        num_interactions=cfg["num_interactions"], num_gaussians=cfg["num_gaussians"],
        cutoff=cfg["cutoff"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], loss=cfg["loss"],
    )
    y_train = torch.tensor([d.y.item() for d in train_ds])
    model.set_target_stats(y_train.mean(), y_train.std())

    ckpt = ModelCheckpoint(monitor="val_loss", save_top_k=3, save_last=True,
                           dirpath=cfg["ckpt_dir"])
    callbacks = [ckpt, EarlyStopping(monitor="val_loss", patience=cfg["patience"])]
    tb_logger = TensorBoardLogger(save_dir="logs", name=cfg["run_name"])

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu", devices=1, precision="16-mixed",
        gradient_clip_val=10.0,
        callbacks=callbacks, logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=cfg.get("progress_bar", True),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
               if k.startswith("test_")}
    n_params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM peak: %.2f GB | params: %d", vram, n_params)

    model.eval().cuda()
    embeddings, emb_y = [], []
    for batch in test_loader:
        batch = batch.cuda()
        embeddings.append(model.embed(batch).cpu().numpy())
        emb_y.extend(batch.y.view(-1).cpu().tolist())

    return {
        "model": model,
        "trainer": trainer,
        "metrics": metrics,
        "y_true": np.array(model.test_true),
        "y_pred": np.array(model.test_pred),
        "embeddings": np.concatenate(embeddings, axis=0),
        "embeddings_y": np.array(emb_y),
        "n_params": n_params,
        "vram_peak_gb": round(vram, 3),
        "best_ckpt": ckpt.best_model_path,
    }
