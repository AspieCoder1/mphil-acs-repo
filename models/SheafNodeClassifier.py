from typing import Literal

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data

from .NodeClassifier import NodeClassifier, CommonStepOutput


class SheafNodeClassifier(NodeClassifier):
    def __init__(self, model: nn.Module, hidden_channels: int = 256,
                 out_channels: int = 10,
                 target: str = "author", task: Literal[
                "binary", "multiclass", "multilabel"] = "multilabel",
                 homogeneous_model: bool = False):
        super().__init__(model=model,
                         hidden_channels=hidden_channels,
                         out_channels=out_channels, target=target,
                         task=task,
                         homogeneous_model=homogeneous_model)
        self.save_hyperparameters()

    def common_step(self, batch: Data, mask: torch.Tensor) -> CommonStepOutput:
        if self.task == "multilabel":
            target_mask = torch.any(~batch.y.isnan(), dim=1)
        else:
            target_mask = batch.y != -1

        mask = torch.logical_and(target_mask, mask)
        y = batch.y[mask]
        logits = self.encoder(batch.x, batch.edge_index)

        y_hat = self.decoder(logits)[mask]

        loss = self.loss_fn(y_hat, y)
        y_hat = self.act_fn(y_hat)
        y = y.to(torch.int)

        return CommonStepOutput(y=y, y_hat=y_hat, loss=loss)

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.train_mask)

        self.train_acc(y_hat, y)
        self.log('train/accuracy', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.val_mask)

        self.val_acc(y_hat, y)

        self.log('valid/accuracy', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('valid/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=1)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.test_mask)

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
                 on_epoch=True, batch_size=1)
        return None
