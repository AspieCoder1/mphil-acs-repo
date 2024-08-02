#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Literal, NamedTuple, TypedDict, Optional

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroDictLinear
import torch.nn.functional as F

from models.sheaf_gnn.transductive.disc_models import DiscreteSheafDiffusion
from .node_classifier import NodeClassifier


class SheafNCSStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor
    maps: torch.Tensor


class TrainStepOutput(TypedDict):
    loss: torch.Tensor
    restriction_maps: torch.Tensor


class SheafNodeClassifier(NodeClassifier):
    def __init__(
        self,
        model: DiscreteSheafDiffusion,
        out_channels: int = 10,
        target: str = "author",
        task: Literal["binary", "multiclass", "multilabel"] = "multilabel",
        in_channels: Optional[dict[str, int]] = None,
        in_feat: int = 64,
        homogeneous_model: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__(
            model=model,
            hidden_channels=model.hidden_dim,
            out_channels=out_channels,
            target=target,
            task=task,
            homogeneous_model=homogeneous_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.save_hyperparameters(ignore=["model"])
        self.fc = HeteroDictLinear(in_channels=in_channels,
                                   out_channels=in_feat)

    def preprocess(self, data: HeteroData) -> (Tensor, Tensor, Tensor):
        x_dict = self.fc(data.x_dict)
        x = F.elu(torch.cat(tuple(x_dict.values()), dim=0))

        return x, data.node_type, data.edge_type

    def common_step(self, batch: HeteroData, step: str = 'train') -> SheafNCSStepOutput:
        x, node_types, edge_types = self.preprocess(batch)
        mask = batch[self.target][f'{step}_mask']

        if self.task == "multilabel":
            mask = torch.any(~batch[self.target].y.isnan(), dim=1)

        y = batch[self.target].y[mask]

        data = Data(x=x, node_type=node_types, edge_type=edge_types)

        logits, maps = self.encoder(data)

        offset = batch.node_offsets[self.target]

        y_hat = self.decoder(logits)[offset:offset + batch[self.target].x.shape[0]][
            mask]

        loss = self.loss_fn(y_hat, y)
        y_hat = self.act_fn(y_hat)
        y = y.to(torch.int)

        return SheafNCSStepOutput(y=y, y_hat=y_hat, loss=loss, maps=maps)

    def training_step(self, batch: HeteroData, batch_idx: int) -> TrainStepOutput:
        y, y_hat, loss, maps = self.common_step(batch, 'train')

        output = self.train_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log(
            "train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1
        )
        return TrainStepOutput(
            loss=loss,
            restriction_maps=maps,
        )

    def validation_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, _ = self.common_step(batch, 'val')

        output = self.valid_metrics(y_hat, y)

        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.log(
            "valid/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, _ = self.common_step(batch, 'test')

        output = self.test_metrics(y_hat, y)
        self.log_dict(
            output, prog_bar=False, on_step=False, on_epoch=True, batch_size=128
        )
        self.log(
            "test/loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        return loss
