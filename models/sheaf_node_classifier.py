from typing import Literal, Dict, Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch_geometric.data import Data


from core.models import get_sheaf_model
from .node_classifier import NodeClassifier, CommonStepOutput


class SheafNodeClassifier(NodeClassifier):
    def __init__(self, model: nn.Module,
                 out_channels: int = 10,
                 target: str = "author", task: Literal[
                "binary", "multiclass", "multilabel"] = "multilabel",
                 homogeneous_model: bool = False):

        super().__init__(model=model,
                         hidden_channels=model.hidden_dim,
                         out_channels=out_channels, target=target,
                         task=task,
                         homogeneous_model=homogeneous_model)
        self.save_hyperparameters(ignore=["model"])

    def common_step(self, batch: Data, mask: torch.Tensor) -> CommonStepOutput:
        if self.task == "multilabel":
            target_mask = torch.any(~batch.y.isnan(), dim=1)
        else:
            target_mask = batch.y != -1

        mask = torch.logical_and(target_mask, mask)
        y = batch.y[mask]
        logits = self.encoder(batch.x)

        y_hat = self.decoder(logits)[mask]

        loss = self.loss_fn(y_hat, y)
        y_hat = self.act_fn(y_hat)
        y = y.to(torch.int)

        return CommonStepOutput(y=y, y_hat=y_hat, loss=loss)

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.train_mask)

        output = self.train_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False,
                      on_epoch=True, batch_size=1)
        self.log('train/loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.val_mask)

        output = self.valid_metrics(y_hat, y)

        self.log_dict(output, prog_bar=True, on_step=False,
                      on_epoch=True, batch_size=1)
        self.log('valid/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=1)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, batch.test_mask)

        output = self.test_metrics(y_hat, y)
        self.log_dict(output, prog_bar=False, on_step=False,
                      on_epoch=True, batch_size=128)
        self.log('test/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=128)

        return loss
