#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Literal, NamedTuple, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HANConv
from torchmetrics.classification import Accuracy, F1Score, AUROC


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor


class HAN(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 256,
        in_channels: Optional[dict[str, int]] = None,
    ):
        super().__init__()

        if in_channels is None:
            in_channels = -1

        self.conv = nn.ModuleList(
            [
                HANConv(
                    in_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                ),
                HANConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                ),
                HANConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                ),
            ]
        )

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        for layer in self.conv:
            x_dict = layer(x_dict, data.edge_index_dict)

        return x_dict


class HANEdgeDecoder(torch.nn.Module):
    def __init__(
        self,
        target: tuple[str, str, str],
    ):
        super().__init__()

        self.rel_src = target[0]
        self.rel_dst = target[-1]

    def forward(self, x_dict, edge_label_index):
        A = x_dict[self.rel_src][edge_label_index[0]]
        B = x_dict[self.rel_dst][edge_label_index[1]]
        return torch.bmm(A.unsqueeze(dim=1), B.unsqueeze(dim=2)).squeeze()


class HANLinkPredictor(L.LightningModule):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 128,
        edge_target: tuple[str, str, str] = ("user", "rates", "movie"),
        in_channels: Optional[dict[str, int]] = None,
    ):
        super(HANLinkPredictor, self).__init__()

        self.target = edge_target
        self.encoder = HAN(metadata, hidden_channels, in_channels)
        self.decoder = HANEdgeDecoder(target=edge_target)

        # metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")

    def common_step(self, batch, pos_idx: str, neg_idx: str) -> CommonStepOutput:
        x_dict = self.encoder(batch)
        num_pos_ex = batch[self.target][pos_idx].size(1)
        num_neg_ex = batch[self.target][neg_idx].size(1)
        neg_ex = torch.randperm(num_neg_ex)[:num_pos_ex]
        neg_samples = batch[self.target][neg_idx][:, neg_ex]
        edge_label_index = torch.hstack((batch[self.target][pos_idx], neg_samples))
        y_hat = self.decoder(x_dict, edge_label_index)
        y = torch.hstack(
            (
                torch.ones(num_pos_ex),
                torch.zeros(num_pos_ex),
            )
        ).to(y_hat)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = torch.round(torch.sigmoid(y_hat))
        return CommonStepOutput(y, y_hat, loss)

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch, "edge_index", "train_neg_edge_index")

        self.train_acc(y_hat, y)

        self.log_dict(
            {
                "train/accuracy": self.train_acc,
                "train/loss": loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(
            batch, "val_pos_edge_index", "val_neg_edge_index"
        )

        self.val_acc(y_hat, y)

        self.log_dict(
            {"valid/accuracy": self.val_acc, "valid/loss": loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(
            batch, "test_pos_edge_index", "test_neg_edge_index"
        )

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)

        self.log_dict(
            {
                "test/accuracy": self.test_acc,
                "test/f1-score": self.test_f1,
                "test/auroc": self.test_auroc,
                "test/loss": loss,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
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
