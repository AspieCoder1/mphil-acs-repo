import functools
from typing import Literal, NamedTuple, Callable

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch_geometric.data import Batch
from torch_geometric.nn import Linear
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor


class NodeClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, hidden_channels: int = 256,
                 out_channels: int = 10,
                 target: str = "author", task: Literal[
                "binary", "multiclass", "multilabel"] = "multilabel",
                 homogeneous_model: bool = False):
        super().__init__()
        self.encoder = model
        self.decoder = Linear(hidden_channels, out_channels)
        self.homogeneous = homogeneous_model

        metrics_params = {
            "task": task,
            "num_labels": out_channels,
            "num_classes": out_channels
        }

        self.train_metrics = MetricCollection([
            F1Score(average="micro", **metrics_params),
            F1Score(average="macro", **metrics_params),
            Accuracy(**metrics_params),
            AUROC(**metrics_params)
        ], prefix="train/")
        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.target = target
        self.task = task

        if task == "multilabel":
            self.loss_fn: Callable = F.multilabel_soft_margin_loss
            self.act_fn: Callable = F.sigmoid
        else:
            self.loss_fn: Callable = F.cross_entropy
            self.act_fn: Callable = functools.partial(F.softmax, dim=-1)

    def common_step(self, batch: Batch, mask: torch.Tensor) -> CommonStepOutput:
        y: torch.Tensor = batch[self.target].y[mask]
        if self.homogeneous:
            x_dict = self.encoder(batch.x_dict, batch.edge_index_dict)
        else:
            x_dict = self.encoder(batch)

        y_hat = self.decoder(x_dict[self.target])[mask]
        loss = self.loss_fn(y_hat, y)
        y_hat = self.act_fn(y_hat)
        y = y.to(torch.int)

        return CommonStepOutput(y, y_hat, loss)

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target].train_mask
        y, y_hat, loss = self.common_step(batch, mask)

        output = self.train_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False,
                      on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=64)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target].val_mask
        y, y_hat, loss = self.common_step(batch, mask)

        output = self.valid_metrics(y_hat, y)

        self.log_dict(output, prog_bar=True, on_step=False,
                      on_epoch=True)
        self.log('valid/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=64)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target].test_mask
        y, y_hat, loss = self.common_step(batch, mask)

        output = self.test_metrics(y_hat, y)
        self.log_dict(output, prog_bar=False, on_step=False,
                      on_epoch=True, batch_size=1)
        self.log('test/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=1)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimiser = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=1_000,
                                                               eta_min=1e-6)

        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss",
            }
        }
