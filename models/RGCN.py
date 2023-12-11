from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.nn import RGCNConv
from torchmetrics import F1Score, AUROC
from torchmetrics.classification import Accuracy


class RGCN(nn.Module):
    def __init__(self, hidden_channels: int, num_classes: int, num_nodes: int,
                 num_relations: int, target: str = "author"):
        super().__init__()
        self.conv1 = RGCNConv(num_nodes, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.linear = nn.Linear(hidden_channels, num_classes)
        self.target = target

    def forward(self, data: HeteroData):
        node_type_names = data.node_types
        data = data.to_homogeneous()

        edge_type, edge_index = data.edge_type, data.edge_index
        x = self.conv1(None, edge_index, edge_type)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(x)

        data.update({
            "x": x
        })
        data = data.to_heterogeneous(node_type_names=node_type_names)
        x = self.linear(data.x_dict[self.target])
        return x


class RGCNNodeClassifier(L.LightningModule):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]],
                 hidden_channels: int, out_channels: int, num_nodes: int,
                 num_relations: int,
                 task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
                 target: str = "paper"):
        super().__init__()
        self.model = RGCN(hidden_channels, out_channels, num_nodes, num_relations,
                          target=target)
        self.num_classes = out_channels

        metrics_params = {
            "task": task,
            "num_labels": out_channels,
            "num_classes": out_channels
        }
        self.train_acc = Accuracy(**metrics_params)
        self.val_acc = Accuracy(**metrics_params)
        self.test_acc = Accuracy(**metrics_params)
        self.test_f1 = F1Score(**metrics_params)
        self.test_auroc = AUROC(**metrics_params)
        self.target_type = target
        self.task = task
        if task == "multilabel":
            self.loss_fn = F.multilabel_soft_margin_loss
        else:
            self.loss_fn = F.cross_entropy

    def common_step(self, batch: Batch, mask: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        y: torch.Tensor = batch[self.target_type].y[mask]
        y_hat = self.model(batch)[mask]
        loss = self.loss_fn(y_hat, y)
        if self.task == "multilabel":
            y_hat = torch.sigmoid(y_hat)
            y = y.to(torch.int)
        else:
            y_hat = y_hat.softmax(dim=-1)
        return y, y_hat, loss

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].train_mask
        y, y_hat, loss = self.common_step(batch, mask)

        self.train_acc(y_hat, y)
        self.log('train/accuracy', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True, batch_size=64)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].val_mask
        y, y_hat, loss = self.common_step(batch, mask)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)

        self.log('valid/accuracy', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('valid/loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, batch_size=64)
        return

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        mask = batch[self.target_type].test_mask
        y, y_hat, loss = self.common_step(batch, mask)

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
                 on_epoch=True, batch_size=128)

        return None

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
