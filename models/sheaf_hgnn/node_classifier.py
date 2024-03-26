#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch_geometric.data import Data
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
)

from .models import SheafHyperGNN, SheafHyperGCN


class SheafHyperGNNNodeClassifier(L.LightningModule):
    def __init__(self, model: Union[SheafHyperGNN, SheafHyperGCN], num_classes):
        super().__init__()

        self.model = model
        self.train_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_classes),
                "F1": MulticlassF1Score(num_classes=num_classes),
                "auroc": MulticlassAUROC(num_classes=num_classes),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def common_step(self, batch: Data, mask: torch.Tensor):
        y = batch.y[mask]
        logits = self.model(batch)[mask]
        loss = F.cross_entropy(logits, y)
        y_hat = F.softmax(logits)
        return loss, y, y_hat

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, y, y_hat = self.common_step(batch, batch.train_mask)
        metrics = self.train_metrics(y_hat, y)

        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=1)
        self.log(
            "train/loss", loss, on_epoch=True, on_step=True, batch_size=1, prog_bar=True
        )

        return loss

    def validation_step(self, batch: Data, batch_idx) -> STEP_OUTPUT:
        loss, y, y_hat = self.common_step(batch, batch.val_mask)
        metrics = self.val_metrics(y_hat, y)

        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=1)
        self.log(
            "val/loss", loss, on_epoch=True, on_step=True, batch_size=1, prog_bar=True
        )

        return loss

    def test_step(self, batch: Data, batch_idx) -> STEP_OUTPUT:
        loss, y, y_hat = self.common_step(batch, batch.test_mask)
        metrics = self.test_metrics(y_hat, y)

        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=1)
        self.log(
            "test/loss", loss, on_epoch=True, on_step=True, batch_size=1, prog_bar=True
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimiser = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=1_000, eta_min=1e-6
        )

        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss",
            },
        }
