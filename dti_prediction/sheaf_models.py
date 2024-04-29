#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

from models.sheaf_hgnn.config import SheafHGNNConfig
from models.sheaf_hgnn.models import SheafHyperGNN
from models.vshae.vshae import VSHAE


class DTIPredictionModule(L.LightningModule):
    def __init__(
        self, model: nn.Module, use_score_function: bool = False, out_channels: int = 64
    ):
        super(DTIPredictionModule, self).__init__()
        self.encoder = model

        self.decoder = None
        if use_score_function:
            self.decoder = nn.Linear(2 * out_channels, 1)

        self.train_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "AUROC": BinaryAUROC(),
                "AUPR": BinaryAveragePrecision(),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def common_step(self, data: Data, pos_idx: Tensor):
        neg_idx = negative_sampling(pos_idx, num_nodes=(data.num_nodes, data.num_nodes))
        pos_neg_idx = torch.column_stack([pos_idx, neg_idx]).to(torch.long)

        logits = self.encoder(data)

        if self.decoder is not None:
            x_cat = torch.cat(
                [logits[pos_neg_idx[0, :]], logits[pos_neg_idx[1, :]]], dim=-1
            )
            preds = self.decoder(x_cat).squeeze()
        else:
            preds = (
                (logits[pos_neg_idx[0]] * logits[pos_neg_idx[1]]).sum(dim=-1)
            )

        targets = torch.hstack(
            (torch.ones(pos_idx.shape[1]), torch.zeros(neg_idx.shape[1]))
        )

        if hasattr(self.encoder, "loss"):
            loss = self.encoder.loss(preds, targets)
        else:
            loss = F.binary_cross_entropy_with_logits(preds, targets)
        return loss, preds, targets.to(torch.long)

    def training_step(self, batch: Data, batch_idx) -> STEP_OUTPUT:
        train_idx = batch.train_idx
        loss, preds, targets = self.common_step(batch, train_idx)
        train_metrics = self.train_metrics(preds, targets)

        self.log_dict(
            train_metrics, prog_bar=False, on_epoch=True, on_step=False, batch_size=1
        )
        self.log(
            "train/loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=1
        )

        return loss

    def test_step(self, batch: Data, batch_idx) -> STEP_OUTPUT:
        train_idx = batch.test_idx
        loss, preds, targets = self.common_step(batch, train_idx)
        test_metrics = self.test_metrics(preds, targets)

        self.log_dict(
            test_metrics, prog_bar=False, on_epoch=True, on_step=False, batch_size=1
        )
        self.log(
            "test/loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=1
        )

        return None

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


class SheafHyperGNNModule(L.LightningModule):
    def __init__(self, args: SheafHGNNConfig, sheaf_type: str):
        super(SheafHyperGNNModule, self).__init__()
        self.model = SheafHyperGNN(args=args, sheaf_type=sheaf_type)
        self.score_func = nn.Linear(2 * self.model.out_dim, 1)

        self.train_metrics = MetricCollection(
            {
                "AUROC": BinaryAUROC(),
                "AUPR": BinaryAveragePrecision(),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def common_step(self, data: Data, pos_idx: Tensor):
        neg_idx = negative_sampling(
            pos_idx, num_nodes=(data.num_nodes, data.num_hyperedges)
        )

        pos_neg_idx = torch.cat([pos_idx, neg_idx])
        logits = self.model(data)

        x_cat = torch.cat([logits[pos_neg_idx[0]], logits[pos_neg_idx[1]]], dim=-1)
        preds = self.score_func(x_cat)
        targets = torch.cat(
            [torch.ones(pos_idx.shape[0]), torch.zeros(neg_idx.shape[0])]
        )

        loss = F.binary_cross_entropy_with_logits(preds, targets)

        return loss, preds, targets

    def training_step(self, batch: Data, batch_idx):
        train_idx = batch.train_idx
        loss, preds, targets = self.common_step(batch, train_idx)
        train_metrics = self.train_metrics(preds, targets)

        self.log_dict(
            train_metrics, prog_bar=False, on_epoch=True, on_step=False, batch_size=1
        )
        self.log(
            "train/loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=1
        )


class VSHAEModule(L.LightningModule):
    def __init__(self, args: SheafHGNNConfig, sheaf_type: str):
        super(VSHAEModule, self).__init__()
        self.model = VSHAE(args=args, sheaf_type=sheaf_type)

        self.train_metrics = MetricCollection(
            {
                "AUROC": BinaryAUROC(),
                "AUPR": BinaryAveragePrecision(),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def common_step(self, data: Data, pos_idx: Tensor):
        neg_idx = negative_sampling(
            pos_idx, num_nodes=(data.num_nodes, data.num_hyperedges)
        )

        pos_neg_idx = torch.cat([pos_idx, neg_idx])
        logits = self.model(data)
        preds = (logits[pos_neg_idx[0]].T @ logits[pos_neg_idx[1]]).squeeze()
        targets = torch.cat(
            [torch.ones(pos_idx.shape[0]), torch.zeros(neg_idx.shape[0])]
        )

        loss = self.model.loss(preds, targets)

        return loss, preds, targets

    def training_step(self, batch: Data, batch_idx):
        train_idx = batch.train_idx
        loss, preds, targets = self.common_step(batch, train_idx)
        train_metrics = self.train_metrics(preds, targets)

        self.log_dict(
            train_metrics, prog_bar=False, on_epoch=True, on_step=False, batch_size=1
        )
        self.log(
            "train/loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=1
        )

        return loss

    def test_step(self, batch: Data, batch_idx):
        test_idx = batch.test_idx
        loss, preds, targets = self.common_step(batch, test_idx)
        test_metrics = self.test_metrics(preds, targets)

        self.log_dict(
            test_metrics, prog_bar=False, on_epoch=True, on_step=False, batch_size=1
        )
        self.log(
            "test/loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=1
        )

        return None
