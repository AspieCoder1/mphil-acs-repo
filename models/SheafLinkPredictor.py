import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torchmetrics.classification import Accuracy, AUROC, F1Score
from torchmetrics.collections import MetricCollection

from models.NodeClassifier import CommonStepOutput


class EdgeDecoder(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(EdgeDecoder, self).__init__()
        self.lin = nn.Linear(2 * in_dims, out_dims)

    def forward(self, x, edge_index):
        h_src = x[edge_index[0]]
        h_dest = x[edge_index[1]]
        concat = torch.concat((h_src, h_dest), dim=1)

        return self.lin(concat)


class SheafLinkPredictor(L.LightningModule):
    def __init__(self, model: nn.Module, batch_size: int = 1,
                 hidden_dim: int = 64, num_classes: int = 1):
        super(SheafLinkPredictor, self).__init__()
        self.encoder = model
        self.batch_size = batch_size
        self.decoder = EdgeDecoder(hidden_dim, num_classes)

        self.train_metrics = MetricCollection({
            "accuracy": Accuracy(task="binary"),
            "auroc": AUROC(task="binary"),
            "f1": F1Score(task="binary")
        }, prefix="train/")

        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.save_hyperparameters()

    def common_step(self, batch: Data) -> CommonStepOutput:
        # (1) Remove NaNs from edge_labels
        label_idx = ~batch.edge_label.isnan()
        y = batch.edge_label[label_idx]

        # (2) Compute the hidden representation of nodes
        h = self.encoder(batch.x, batch.edge_index)

        # (3) reduced edge_index
        edge_index = batch.edge_label_index[:, label_idx]

        # (4) Calculate dot product h[i].h[j] for i, j in edge_index
        y_hat = self.decoder(h, edge_index).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)

        return CommonStepOutput(loss=loss, y=y, y_hat=y_hat)

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        outputs = self.train_metrics(y_hat, y)

        self.log_dict(outputs)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        outputs = self.valid_metrics(y_hat, y)

        self.log_dict(outputs)
        self.log("valid/loss", loss)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        outputs = self.test_metrics(y_hat, y)

        self.log_dict(outputs)
        self.log("test/loss", loss)

        return loss

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
