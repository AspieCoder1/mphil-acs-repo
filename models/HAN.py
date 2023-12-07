from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler
)
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import HANConv
from torchmetrics.classification import Accuracy, F1Score, AUROC


class HAN(nn.Module):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]],
                 hidden_channels: int = 256, out_channels: int = 10,
                 target_type: str = "author"):
        super().__init__()
        self.conv = nn.ModuleList([
            HANConv(-1, hidden_channels, heads=8, dropout=0.6,
                    metadata=metadata),
            HANConv(hidden_channels, hidden_channels, heads=8, dropout=0.6,
                    metadata=metadata),
            HANConv(hidden_channels, hidden_channels, heads=8, dropout=0.6,
                    metadata=metadata)
        ]
        )
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.target_type = target_type

    def forward(self, data: Data):
        x_dict = data.x_dict
        for layer in self.conv:
            x_dict = layer(x_dict, data.edge_index_dict)

        out = self.linear(x_dict[self.target_type])
        return out


class HANEntityPredictor(L.LightningModule):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]],
                 hidden_channels: int = 128, out_channels: int = 10,
                 target: str = "author", task: Literal[
                "binary", "multiclass", "multilabel"] = "multilabel"):
        super().__init__()
        self.model = HAN(metadata, hidden_channels, out_channels, target)
        self.num_classes = out_channels

        metrics_params = {
            "task": task,
            "num_labels": out_channels,
            "num_classes": out_channels
        }
        self.train_acc = Accuracy(**metrics_params)
        self.val_acc = Accuracy(**metrics_params)
        self.test_acc = Accuracy(**metrics_params)
        self.test_f1 = F1Score(**metrics_params)
        self.test_auroc = AUROC(**metrics_params)
        self.target_type = target
        self.task = task
        if task == "multilabel":
            self.loss_fn = F.multilabel_soft_margin_loss
        else:
            self.loss_fn = F.cross_entropy

    def common_step(self, batch: Batch, mask: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        y: torch.Tensor = batch[self.target_type].y[mask]
        y = y.to(torch.int)
        y_hat = self.model(batch)[mask]
        loss = self.loss_fn(y_hat, y)
        if self.task == "multilabel":
            y_hat = torch.sigmoid(y_hat)
        else:
            y_hat = y_hat.softmax(dim=-1)
        return y, y_hat, loss

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].train_mask
        y, y_hat, loss = self.common_step(batch, mask)

        self.train_acc(y_hat, y)
        self.log('train/accuracy', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=64)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].val_mask
        y, y_hat, loss = self.common_step(batch, mask)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)

        self.log('valid/accuracy', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('valid/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=64)
        return

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].test_mask
        y, y_hat, loss = self.common_step(batch, mask)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)

        self.log('test/accuracy', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('test/f1-score', self.test_f1, prog_bar=False, on_step=False,
                 on_epoch=True)
        self.log('test/auroc', self.test_auroc, prog_bar=False, on_step=False,
                 on_epoch=True)
        self.log('test/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=128)

        return None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimiser = torch.optim.AdamW(self.parameters(), lr=0.005, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.9)

        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss",
            }
        }
