import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, HeteroData

from models import LinkPredictor
from models.NodeClassifier import CommonStepOutput


class SheafEdgeDecoder(nn.Module):
    def __init__(self):
        super(SheafEdgeDecoder, self).__init__()

    def forward(self, x, edge_index):
        h_src = x[edge_index[0]]
        h_dest = x[edge_index[1]]

        return torch.bmm(h_src.unsqueeze(dim=1), h_dest.unsqueeze(dim=2)).squeeze()


class SheafLinkPredictor(LinkPredictor):
    def __init__(self, model: nn.Module,
                 edge_target: tuple[str, str, str] = ("user", "rates", "movie"),
                 homogeneous: bool = False,
                 batch_size: int = 1,
                 decoder=SheafEdgeDecoder()):
        super(LinkPredictor).__init__(
            model=model,
            edge_target=edge_target,
            homogeneous=homogeneous,
            batch_size=batch_size
        )

        self.decoder = decoder

        self.save_hyperparameters()

    def common_step(self, batch: Data) -> CommonStepOutput:
        # (1) Remove NaNs from edge_labels
        label_idx = ~batch.edge_label.is_nan()
        y = batch.edge_label[label_idx]

        # (2) Compute the hidden representation of nodes
        h = self.encoder(batch.x, batch.edge_index)

        # (3) reduced edge_index
        edge_index = batch.edge_index[label_idx]

        # (4) Calculate dot product h[i].h[j] for i, j in edge_index
        y_hat = self.decoder(h, edge_index)
        print(y_hat)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = F.sigmoid(y_hat)

        return CommonStepOutput(loss=loss, y=y, y_hat=y_hat)

    def training_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.train_acc(y_hat, y)

        self.log_dict({
            "train/accuracy": self.train_acc,
            "train/loss": loss,
        })

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.val_acc(y_hat, y)

        self.log_dict({
            "valid/accuracy": self.val_acc,
            "valid/loss": loss
        })

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(batch)

        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log_dict({
            "test/accuracy": self.test_acc,
            "test/f1-score": self.test_f1,
            "test/auroc": self.test_auroc,
            "test/loss": loss
        })

        return loss
