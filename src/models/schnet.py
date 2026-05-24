from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SchNet
from torch_geometric.utils import scatter
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score


class LitSchNet(pl.LightningModule):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50, cutoff: float = 6.0,
                 lr: float = 1e-3, weight_decay: float = 0.0, loss: str = "l1"):
        super().__init__()
        self.save_hyperparameters()
        self.net = SchNet(
            hidden_channels=hidden_channels, num_filters=num_filters,
            num_interactions=num_interactions, num_gaussians=num_gaussians, cutoff=cutoff,
        )
        self.loss_fn = F.l1_loss if loss == "l1" else F.mse_loss
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

        self.val_r2, self.val_mae, self.val_mse = R2Score(), MeanAbsoluteError(), MeanSquaredError()
        self.test_r2, self.test_mae, self.test_mse = R2Score(), MeanAbsoluteError(), MeanSquaredError()
        self.test_true: list[float] = []
        self.test_pred: list[float] = []

    def set_target_stats(self, mean: float, std: float) -> None:
        self.y_mean = torch.tensor(float(mean))
        self.y_std = torch.tensor(float(std))

    def forward(self, data) -> torch.Tensor:
        h = self.net.embedding(data.z)
        edge_attr = self.net.distance_expansion(data.edge_weight)
        for interaction in self.net.interactions:
            h = h + interaction(h, data.edge_index, data.edge_weight, edge_attr)
        h = self.net.lin1(h)
        h = self.net.act(h)
        h = self.net.lin2(h)
        out = scatter(h, data.batch, dim=0, reduce="sum").view(-1)
        return out * self.y_std + self.y_mean

    @torch.no_grad()
    def embed(self, data) -> torch.Tensor:
        h = self.net.embedding(data.z)
        edge_attr = self.net.distance_expansion(data.edge_weight)
        for interaction in self.net.interactions:
            h = h + interaction(h, data.edge_index, data.edge_weight, edge_attr)
        return scatter(h, data.batch, dim=0, reduce="sum")

    def training_step(self, batch, _idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch.y.view(-1))
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
        self.test_true.extend(y.detach().cpu().tolist())
        self.test_pred.extend(pred.detach().cpu().tolist())

    def on_test_epoch_end(self):
        self.log("test_r2", self.test_r2.compute())
        self.log("test_mae", self.test_mae.compute())
        self.log("test_mse", self.test_mse.compute())
        self.log("test_rmse", self.test_mse.compute().sqrt())
        for m in (self.test_r2, self.test_mae, self.test_mse):
            m.reset()

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
