#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import functools
from typing import Literal, NamedTuple, TypedDict, Optional, Callable

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroDictLinear
from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Accuracy, AUROC
from models.gnn_baselines import GCN

from models.sheaf_gnn.transductive.disc_models import DiscreteSheafDiffusion


class SheafNCSStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor


class TrainStepOutput(TypedDict):
    loss: torch.Tensor
    restriction_maps: torch.Tensor


class SheafNodeClassifier(L.LightningModule):
    def __init__(
        self,
        model: DiscreteSheafDiffusion,
        out_channels: int = 10,
        target: str = "author",
        task: Literal["binary", "multiclass", "multilabel"] = "multilabel",
        in_channels: Optional[dict[str, int]] = None,
        in_feat: int = 64,
        scheduler: Optional[LRSchedulerCallable] = None,
        optimiser: Optional[OptimizerCallable] = None,
    ):
        super().__init__()
        self.encoder = model
        self.decoder = nn.Linear(model.hidden_dim, out_channels)
        self.scheduler: Optional[LRSchedulerCallable] = scheduler
        self.optimiser = optimiser
        self.gcn = GCN(in_channels=in_feat, hidden_channels=model.hidden_dim)

        metrics_params = {
            "task": task,
            "num_labels": out_channels,
            "num_classes": out_channels,
        }

        self.train_metrics = MetricCollection(
            {
                "micro-f1": F1Score(average="micro", **metrics_params),
                "macro-f1": F1Score(average="macro", **metrics_params),
                "accuracy": Accuracy(**metrics_params),
                "auroc": AUROC(**metrics_params),
            },
            prefix="train/",
        )
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

        self.save_hyperparameters(ignore=["model"])
        self.fc = HeteroDictLinear(in_channels=in_channels,
                                   out_channels=in_feat)

    def common_step(self, batch: HeteroData, step: str = 'train') -> SheafNCSStepOutput:
        x_dict = self.fc(batch.x_dict)
        x = F.elu(torch.cat(tuple(x_dict.values()), dim=0))

        mask = batch[self.target][f'{step}_mask']

        if self.task == "multilabel":
            mask = torch.any(~batch[self.target].y.isnan(), dim=1)

        y = batch[self.target].y[mask]
        # logits = self.gcn(x, batch.homo_edge_index)
        # logits = F.normalize(logits, dim=1, p=2)
        logits = self.encoder(x, batch.node_type, batch.edge_type)


        offset = batch.node_offsets[self.target]

        y_hat = self.decoder(logits)[offset:offset + batch[self.target].x.shape[0]][
            mask]

        loss = self.loss_fn(y_hat, y)
        y_hat = self.act_fn(y_hat)
        y = y.to(torch.int)

        return SheafNCSStepOutput(y=y, y_hat=y_hat, loss=loss)

    def training_step(self, batch: HeteroData, batch_idx: int) -> TrainStepOutput:
        y, y_hat, loss = self.common_step(batch, 'train')

        output = self.train_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log(
            "train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1
        )
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, 'val')

        output = self.valid_metrics(y_hat, y)

        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log(
            "valid/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, 'test')

        output = self.test_metrics(y_hat, y)
        self.log_dict(
            output, prog_bar=False, on_step=False, on_epoch=True, batch_size=128
        )
        self.log(
            "test/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimiser = self.optimiser(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimiser)
            return {
                "optimizer": optimiser,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/micro-f1",
                },
            }
        return optimiser
