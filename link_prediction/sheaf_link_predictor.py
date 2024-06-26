#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Callable, NamedTuple

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
)
from torchmetrics.collections import MetricCollection
from torchmetrics.retrieval import RetrievalMRR


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor
    indexes: torch.Tensor


class SheafLinkPredictor(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 1,
    ):
        super(SheafLinkPredictor, self).__init__()
        self.encoder = model
        self.batch_size = batch_size
        self.decoder = nn.Linear(2*hidden_dim, num_classes)

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

        self.save_hyperparameters(ignore="model")

    def common_step(self, batch: Data) -> CommonStepOutput:
        # (1) Remove NaNs from edge_labels
        label_idx = ~batch.edge_label.isnan()
        y = batch.edge_label[label_idx]

        # (2) Compute the hidden representation of nodes
        h, _ = self.encoder(batch)

        # (3) reduced edge_label_index
        edge_label_index = batch.edge_label_index[:, label_idx]

        # (4) Calculate dot product h[i].h[j] for i, j in edge_label_index
        h_src = h[edge_label_index[0, :]]
        h_dest = h[edge_label_index[1, :]]
        y_hat = self.decoder(torch.concat((h_src, h_dest), dim=1)).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)

        return CommonStepOutput(loss=loss, y=y.to(torch.long), y_hat=y_hat,
                                indexes=edge_label_index[0, :])

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        metrics = self.train_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        metrics = self.valid_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("valid/loss", loss)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        metrics = self.test_metrics(preds=y_hat, target=y, indexes=indexes)

        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("test/loss", loss)

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
