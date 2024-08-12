#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Callable, NamedTuple, Literal, Optional

import lightning as L
import torch
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroDictLinear
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
)
from torchmetrics.collections import MetricCollection
from torchmetrics.retrieval import RetrievalMRR

from link_prediction.decoders import LinkPredDecoder


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor
    indexes: torch.Tensor


class SheafLinkPredictor(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        target: tuple[str, str, str],
        in_channels: Optional[dict[str, int]] = None,
        in_feat: int = 64,
        batch_size: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 1,
        scheduler: Optional[LRSchedulerCallable] = None,
        optimiser: Optional[OptimizerCallable] = None,
        decoder: Optional[LinkPredDecoder] = None
    ):
        super(SheafLinkPredictor, self).__init__()
        self.encoder = model
        self.batch_size = batch_size
        self.decoder = decoder
        self.target = target
        self.fc = HeteroDictLinear(in_channels=in_channels,
                                   out_channels=in_feat)

        self.train_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "AUROC": BinaryAUROC(),
                "AUPR": BinaryAveragePrecision(),
                "MRR": RetrievalMRR()
            },
            prefix="train/",
        )

        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")
        self.loss_fn: Callable = F.binary_cross_entropy_with_logits
        self.scheduler = scheduler
        self.optimiser = optimiser

        self.save_hyperparameters(ignore="model")

    def preprocess(self, data: HeteroData) -> Tensor:
        x_dict = self.fc(data.x_dict)
        x = F.elu(torch.cat(tuple(x_dict.values()), dim=0))

        return x

    def common_step(self, batch: HeteroData,
                    stage: Literal['train', 'val', 'test']) -> CommonStepOutput:
        edge_label = batch[self.target][f'{stage}_edge_label']
        edge_label_index = batch[self.target][f'{stage}_edge_label_index']
        x = self.preprocess(batch)

        # (2) Compute the hidden representation of nodes
        h = self.encoder(x, batch.node_type, batch.edge_type)

        # (4) Calculate dot product h[i].h[j] for i, j in edge_label_index
        h_src = h[batch.node_offsets[self.target[0]] + edge_label_index[0, :]]
        h_dest = h[batch.node_offsets[self.target[-1]] + edge_label_index[1, :]]
        y_hat = self.decoder(h_src, h_dest).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, edge_label.to(torch.float))
        y_hat = F.sigmoid(y_hat)

        return CommonStepOutput(loss=loss, y=edge_label.to(torch.long), y_hat=y_hat,
                                indexes=edge_label_index[0, :].to(torch.long))

    def training_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch, 'train')

        metrics = self.train_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=True, on_step=False, on_epoch=True, batch_size=1
        )
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1)

        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch, 'val')

        metrics = self.valid_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1)
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch, 'test')

        metrics = self.test_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("test/loss", loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=1)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimiser = self.optimiser(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimiser)
            return {
                "optimizer": optimiser,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/AUROC",
                },
            }
        return optimiser


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """

    __constants__ = ["lambda_reg"]
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        log_prob = F.logsigmoid(positives - negatives).mean()
        return -log_prob
