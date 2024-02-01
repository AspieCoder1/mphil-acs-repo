import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torchmetrics.classification import Accuracy, F1Score, AUROC

from models.NodeClassifier import CommonStepOutput


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
    def __init__(self, model: nn.Module,
                 edge_target: tuple[str, str, str] = ("user", "rates", "movie"),
                 homogeneous: bool = False,
                 batch_size: int = 1):
        super(LinkPredictor, self).__init__()
        self.encoder = model
        self.decoder = EdgeDecoder(target=edge_target, hidden_dim=256, out_dim=1)
        self.homogeneous = homogeneous

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.target = edge_target
        self.batch_size = batch_size

    def common_step(self, batch: HeteroData) -> CommonStepOutput:
        if self.homogeneous:
            x_dict = self.encoder(batch.x_dict, batch.edge_index_dict)
        else:
            x_dict = self.encoder(batch)

        y_hat = self.decoder(x_dict,
                             batch[self.target].edge_label_index).flatten()
        y = batch[self.target].edge_label

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        return CommonStepOutput(y, y_hat, loss)

    def training_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.train_acc(y_hat, y)

        self.log_dict(
            {
                "train/accuracy": self.train_acc,
                "train/loss": loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.val_acc(y_hat, y)

        self.log_dict(
            {
                "valid/accuracy": self.val_acc,
                "valid/loss": loss
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)

        self.log_dict(
            {
                "test/accuracy": self.test_acc,
                "test/f1-score": self.test_f1,
                "test/auroc": self.test_auroc,
                "test/loss": loss
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )

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
