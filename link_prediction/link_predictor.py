#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Optional, Union, NamedTuple

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import HeteroData, Data
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryAveragePrecision, )
from torchmetrics.retrieval import RetrievalMRR


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor
    indexes: torch.Tensor


class EdgeDecoder(nn.Module):
    def __init__(
        self,
        target: tuple[str, str, str],
        hidden_dim: int = 64,
        out_dim: int = 1,
    ):
        super().__init__()

        self.rel_src = target[0]
        self.rel_dst = target[-1]
        self.lin = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, x_dict, edge_label_index):
        h_src = x_dict[self.rel_src][edge_label_index[0]]
        h_dest = x_dict[self.rel_dst][edge_label_index[1]]
        concat = torch.concat([h_src, h_dest], dim=1)
        return self.lin(concat)


class LinkPredictor(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        edge_target: tuple[str, str, str] = ("user", "rates", "movie"),
        homogeneous: bool = False,
        batch_size: int = 1,
            node_type_names: Optional[list[str]] = None,
            edge_type_names: Optional[list[tuple[str, str, str]]] = None,
    ):
        super(LinkPredictor, self).__init__()
        self.encoder = model
        self.score_func = nn.Linear(512, 1)
        self.homogeneous = homogeneous
        self.target = edge_target

        self.train_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "AUROC": BinaryAUROC(),
                "AUPR": BinaryAveragePrecision(),
                "MRR": RetrievalMRR(),
            },
            prefix="train/",
        )

        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")
        self.batch_size = batch_size

        if node_type_names and edge_type_names:
            self.src_type = node_type_names.index(edge_target[0])
            self.dst_type = node_type_names.index(edge_target[-1])
            self.edge_type = edge_type_names.index(edge_target)

        self.save_hyperparameters(ignore="model")

    def common_step_homo(self, batch: Data) -> CommonStepOutput:
        label_idx = ~batch.edge_label.isnan()
        y = batch.edge_label[label_idx]
        edge_label_idx = batch.edge_label_index[:, label_idx]

        out = self.encoder(batch)

        h_src = out[edge_label_idx[0]]
        h_dst = out[edge_label_idx[1]]
        h_cat = torch.cat([h_src, h_dst], dim=1)
        y_hat = self.score_func(h_cat).flatten()

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        return CommonStepOutput(y.to(torch.long), y_hat, loss,
                                indexes=edge_label_idx[0])

    def common_step_hetero(self, batch: HeteroData):
        edge_label_idx = batch[self.target].edge_label_index
        y = batch[self.target].edge_label
        out = self.encoder(batch)

        h_src = out[self.target[0]][edge_label_idx[0]]
        h_dst = out[self.target[-1]][edge_label_idx[1]]
        h_cat = torch.cat([h_src, h_dst], dim=1)

        y_hat = self.score_func(h_cat).flatten()

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        return CommonStepOutput(y.to(torch.long), y_hat, loss,
                                indexes=edge_label_idx[0])

    def common_step(self, batch: Union[Data, HeteroData]):
        if isinstance(batch, HeteroData):
            return self.common_step_hetero(batch)
        return self.common_step_homo(batch)

    def training_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        self.log_dict(
            self.train_metrics(preds=y_hat, target=y, indexes=indexes),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log("train/loss", loss, batch_size=1)

        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        self.log_dict(
            self.valid_metrics(preds=y_hat, target=y, indexes=indexes),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("valid/loss", loss, batch_size=1, on_epoch=True)

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, indexes = self.common_step(batch)

        self.log_dict(
            self.test_metrics(preds=y_hat, target=y, indexes=indexes),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("test/loss", loss, batch_size=1)

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
