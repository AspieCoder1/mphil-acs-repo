#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from ..sheaf_hgnn.models import SheafHyperGNN, SheafHGNNConfig, HGNNSheafTypes

EPS = 1e-15
MAX_LOGSTD = 10


class VSHAE(nn.Module):
    def __init__(self, args: SheafHGNNConfig, sheaf_type: HGNNSheafTypes):
        super(VSHAE, self).__init__()
        self.encoder = SheafHyperGNN(args=args, sheaf_type=sheaf_type)
        self.mu_encoder = nn.Linear(self.encoder.out_dim, 128)
        self.logstd_encoder = nn.Linear(self.encoder.out_dim, 128)
        self.mu = nn.Parameter(Tensor([0]))
        self.logstd = nn.Parameter(Tensor([0]))

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.mu_encoder.reset_parameters()
        self.logstd_encoder.reset_parameters()

    def reparametrise(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, data: Data):
        H = self.encoder(data)
        self.mu = F.elu(self.mu_encoder(H))
        self.logstd = F.elu(self.logstd_encoder(H))
        self.logstd = self.logstd.clamp(max=MAX_LOGSTD)
        return self.reparametrise(self.mu, self.logstd)

    def loss(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets)
        kl_divergence = -0.5 * torch.mean(
            torch.sum(
                torch.sum(
                    1 + 2 * self.logstd - self.mu**2 - self.logstd.exp() ** 2, dim=1
                )
            )
        )

        return BCE_loss + kl_divergence


class SheafHyperGNNModule(L.LightningModule):
    def __init__(self, args: SheafHGNNConfig, sheaf_type: HGNNSheafTypes):
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
    def __init__(self, args: SheafHGNNConfig, sheaf_type: HGNNSheafTypes):
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
