"""MACE-MP-0 fine-tune LightningModule for ΔG_H prediction.

Loads the pre-trained MACE-MP-0 model, runs message passing to obtain per-atom
node features, pools [emb(H), mean(emb(vizinhos))], and passes the result through
a small MLP regression head.

Supports three training stages:
  - freeze_backbone=True: only the MLP head is trained
  - freeze_backbone=True, unfreeze_last_n > 0: last N interaction layers + head
  - freeze_backbone=False: full end-to-end fine-tune
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import scatter
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from ase import Atoms
from mace import data as mace_data
from mace.tools import torch_tools

logger = logging.getLogger(__name__)


def _pool_mask(
    node_feats: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool node_feats for nodes where mask is True, per graph."""
    dtype = node_feats.dtype
    mask_f = mask.unsqueeze(-1).to(dtype)
    counts = scatter(mask.to(dtype), batch, dim=0, dim_size=num_graphs).clamp(min=1)
    sums = scatter(node_feats * mask_f, batch, dim=0, dim_size=num_graphs)
    return sums / counts.unsqueeze(-1)


class LitMACEFineTune(pl.LightningModule):
    """LightningModule wrapping a pre-trained MACE model with a ΔG_H regression head."""

    def __init__(
        self,
        # Model loading
        mace_model: str = "medium",
        mace_device: str | None = None,
        default_dtype: str = "float32",
        # Head architecture
        hidden_dim: int = 256,
        head_layers: int = 2,
        dropout: float = 0.0,
        # Freeze strategy
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
        # Optimizer
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "l1",
        # Scheduler
        lr_patience: int = 15,
        lr_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Load MACE model ---
        from mace.calculators import mace_mp

        device = mace_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        dt = dtype_map.get(default_dtype, torch.float32)

        calc = mace_mp(model=mace_model, device=device, default_dtype=default_dtype)
        self.mace = calc.models[0]  # ScaleShiftMACE
        self.r_max = float(self.mace.r_max.item())
        self.z_table = calc.z_table
        self.heads = ["Default"]

        # Node feature dimension from the model
        self.node_feat_dim: int = self._infer_node_dim()

        # --- Freeze/unfreeze ---
        self._apply_freeze_strategy(freeze_backbone, unfreeze_last_n)

        # --- MLP regression head ---
        head_input_dim = self.node_feat_dim * 2  # [emb(H), mean(emb(neighbors))]
        layers: list[nn.Module] = []
        in_dim = head_input_dim
        for i in range(head_layers):
            out_dim = hidden_dim if i < head_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < head_layers - 1:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.head = nn.Sequential(*layers)

        # --- Loss ---
        self.loss_fn = F.l1_loss if loss == "l1" else F.mse_loss

        # --- Target normalization ---
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

        # --- Metrics ---
        self.val_r2 = R2Score()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.test_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_true: list[float] = []
        self.test_pred: list[float] = []

    def _infer_node_dim(self) -> int:
        """Determine node_feats dimension by running a dummy forward pass."""
        dummy = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
        with torch_tools.default_dtype(torch.float32):
            config = mace_data.config_from_atoms(dummy)
            graph = mace_data.AtomicData.from_config(
                config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads,
            )
        n = graph.num_nodes
        d = graph.to_dict()
        d["batch"] = torch.zeros(n, dtype=torch.long)
        d["head"] = d["head"].unsqueeze(0)
        d["ptr"] = torch.tensor([0, n], dtype=torch.long)
        # Move to same device as model
        device = next(self.mace.parameters()).device
        d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        with torch.no_grad():
            out = self.mace(d, compute_force=False, training=False)
        return out["node_feats"].shape[-1]  # e.g. 640 for medium

    def _apply_freeze_strategy(self, freeze_backbone: bool, unfreeze_last_n: int) -> None:
        """Freeze or partially unfreeze the MACE backbone."""
        if not freeze_backbone:
            for p in self.mace.parameters():
                p.requires_grad = True
            logger.info("MACE backbone: fully trainable")
            return

        # Freeze all
        for p in self.mace.parameters():
            p.requires_grad = False
        logger.info("MACE backbone: frozen")

        if unfreeze_last_n > 0:
            # Unfreeze last N interaction blocks and their products/readouts
            n_int = len(self.mace.interactions)
            start = max(0, n_int - unfreeze_last_n)
            for i in range(start, n_int):
                for p in self.mace.interactions[i].parameters():
                    p.requires_grad = True
            for i in range(start, len(self.mace.products)):
                for p in self.mace.products[i].parameters():
                    p.requires_grad = True
            for p in self.mace.readouts.parameters():
                p.requires_grad = True
            logger.info("MACE backbone: unfrozen last %d interaction layers", unfreeze_last_n)

    def set_target_stats(self, mean: float, std: float) -> None:
        self.y_mean = torch.tensor(float(mean))
        self.y_std = torch.tensor(float(std))

    def forward(self, batch: Any) -> torch.Tensor:
        """Predict ΔG_H (denormalized) for a PyG batch of AtomicData graphs."""
        d = batch.to_dict()
        d["ptr"] = batch.ptr
        d["batch"] = batch.batch
        # head should be per-graph - PyG batching may preserve scalar; fix if needed
        if d["head"].ndim == 0:
            d["head"] = d["head"].unsqueeze(0)

        out = self.mace(d, compute_force=False, training=self.training)
        node_feats = out["node_feats"]  # (total_nodes, D)

        num_graphs = int(batch.ptr.numel() - 1)
        emb_h = _pool_mask(node_feats, batch.batch, num_graphs, batch.h_mask)
        emb_n = _pool_mask(node_feats, batch.batch, num_graphs, batch.nb_mask)
        pooled = torch.cat([emb_h, emb_n], dim=-1)  # (num_graphs, 2*D)

        raw = self.head(pooled).view(-1)  # (num_graphs,)
        return raw * self.y_std + self.y_mean

    def training_step(self, batch, _idx):
        pred = self(batch)
        y = batch.y.view(-1)
        y_norm = (y - self.y_mean) / self.y_std
        pred_norm = (pred - self.y_mean) / self.y_std
        loss = self.loss_fn(pred_norm, y_norm)
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, _idx):
        pred, y = self(batch), batch.y.view(-1)
        self.log("val_loss", self.loss_fn(pred, y), batch_size=batch.num_graphs, prog_bar=True)
        self.val_r2.update(pred, y)
        self.val_mae.update(pred, y)
        self.val_mse.update(pred, y)

    def on_validation_epoch_end(self):
        self.log("val_r2", self.val_r2.compute(), prog_bar=True)
        self.log("val_mae", self.val_mae.compute())
        self.log("val_rmse", self.val_mse.compute().sqrt())
        for m in (self.val_r2, self.val_mae, self.val_mse):
            m.reset()
        if self.device.type == "cuda":
            self.log("vram_gb", torch.cuda.max_memory_allocated() / 1e9)

    def test_step(self, batch, _idx):
        pred, y = self(batch), batch.y.view(-1)
        self.test_r2.update(pred, y)
        self.test_mae.update(pred, y)
        self.test_mse.update(pred, y)
        self.test_true.extend(y.cpu().tolist())
        self.test_pred.extend(pred.cpu().tolist())

    def on_test_epoch_end(self):
        self.log("test_r2", self.test_r2.compute())
        self.log("test_mae", self.test_mae.compute())
        self.log("test_rmse", self.test_mse.compute().sqrt())
        for m in (self.test_r2, self.test_mae, self.test_mse):
            m.reset()

    def configure_optimizers(self):
        opt = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = ReduceLROnPlateau(opt, mode="min", factor=self.hparams.lr_factor,
                                  patience=self.hparams.lr_patience)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    @torch.no_grad()
    def embed(self, batch: Any) -> torch.Tensor:
        """Extract pooled graph-level embeddings (for analysis)."""
        d = batch.to_dict()
        d["ptr"] = batch.ptr
        d["batch"] = batch.batch
        if d["head"].ndim == 0:
            d["head"] = d["head"].unsqueeze(0)
        out = self.mace(d, compute_force=False, training=False)
        node_feats = out["node_feats"]
        num_graphs = int(batch.ptr.numel() - 1)
        emb_h = _pool_mask(node_feats, batch.batch, num_graphs, batch.h_mask)
        emb_n = _pool_mask(node_feats, batch.batch, num_graphs, batch.nb_mask)
        return torch.cat([emb_h, emb_n], dim=-1)
