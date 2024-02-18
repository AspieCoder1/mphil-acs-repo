import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalMRR, RetrievalHitRate
)

from models.sheaf_link_predictor import RecSysStepOutput


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
        self.target = edge_target

        self.train_metrics = MetricCollection({
            "nDCG@20": RetrievalNormalizedDCG(top_k=20),
            "recall@20": RetrievalRecall(top_k=20),
            "precision@20": RetrievalPrecision(top_k=20),
            "HR@20": RetrievalHitRate(top_k=20),
            "MRR": RetrievalMRR(top_k=20)
        }, prefix="train/")

        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")
        self.batch_size = batch_size
        self.save_hyperparameters(ignore='model')

    def common_step(self, batch: HeteroData) -> RecSysStepOutput:
        if self.homogeneous:
            x_dict = self.encoder(batch.x_dict, batch.edge_index_dict)
        else:
            x_dict = self.encoder(batch)

        y_hat = self.decoder(x_dict,
                             batch[self.target].edge_label_index).flatten()
        y = batch[self.target].edge_label

        loss = F.binary_cross_entropy_with_logits(y, y_hat)
        y_hat = F.sigmoid(y_hat)
        return RecSysStepOutput(y, y_hat, loss, batch[self.target].edge_label_index[0])

    def training_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, index = self.common_step(batch)

        self.log_dict(
            self.train_metrics(y_hat, y, index),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True
        )
        self.log("train/loss", loss, batch_size=1)

        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, index = self.common_step(batch)

        self.log_dict(
            self.valid_metrics(y_hat, y, index),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )
        self.log("valid/loss", loss, batch_size=1, on_epoch=True)

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, index = self.common_step(batch)

        self.log_dict(
            self.test_metrics(y_hat, y, index),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True
        )
        self.log('test/loss', loss, batch_size=1)

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
