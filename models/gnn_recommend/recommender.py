#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
#
#  Adapted from https://medium.com/stanford-cs224w/spotify-track-neural-recommender-system-51d266e31e16

from typing import Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import Adj, Tensor, OptTensor
from torchmetrics import MetricCollection
from torch_geometric.nn import HeteroDictLinear

from .metrics import LinkPredNDCG, LinkPredRecall

DataOrHeteroData = Union[Data, HeteroData]


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """

    __constants__ = ["lambda_reg"]
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(
        self, positives: Tensor, negatives: Tensor, parameters: Tensor = None
    ) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).sum()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs


class _Recommender(torch.nn.Module):
    """
    Here we adapt the LightGCN model from Torch Geometric for our purposes. We allow
    for customizable convolutional layers, custom embeddings. In addition, we deifne some
    additional custom functions.

    """

    def __init__(
        self,
        encoder: nn.Module,
        target=("user", "rates", "artist"),
        is_hetero: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        self.target = target
        self.is_hetero = is_hetero

    def get_embedding(self, data: DataOrHeteroData) -> Tensor:
        if isinstance(data, HeteroData):
            return self.encoder(data)
        return F.elu(self.encoder(data.x, data.edge_index))

    def forward(self, data: DataOrHeteroData) -> Tensor:
        out = self.get_embedding(data)
        return out

    def predict_link(self, data: DataOrHeteroData, prob: bool = False) -> Tensor:
        pred = self(data).sigmoid()
        return pred if prob else pred.round()

    def predict_link_embedding(self, embed: Adj, edge_label_index: Adj) -> Tensor:
        if self.is_hetero:
            embed_src = embed[self.target[0]][edge_label_index[0]]
            embed_dst = embed[self.target[-1]][edge_label_index[1]]
        else:
            embed_src = embed[edge_label_index[0]]
            embed_dst = embed[edge_label_index[1]]
        return (embed_src * embed_dst).sum(dim=-1)

    def recommend(
        self,
        data: DataOrHeteroData,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
    ) -> Tensor:
        assert src_index is not None and dst_index is not None

        out_src = out_dst = self.get_embedding(data)

        if isinstance(data, HeteroData):
            out_src = out_src[self.target[0]][src_index]
            out_dst = out_dst[self.target[-1]][dst_index]
        else:
            out_src = out_src[src_index]
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        top_index = dst_index[top_index.view(-1)].view(*top_index.size())
        return top_index

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss."""
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, None)

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.num_nodes}, "
            f"{self.embedding_dim}, num_layers={self.num_layers})"
        )


class GNNRecommender(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        edge_target: tuple[str, str, str] = ("user", "rates", "movie"),
        homogeneous: bool = False,
        batch_size: int = 1,
    ):
        super(GNNRecommender, self).__init__()
        self.recommender: _Recommender = _Recommender(model, target=edge_target)
        self.homogeneous = homogeneous
        self.target = edge_target
        self.batch_size = batch_size
        self.train_metrics = MetricCollection(
            {
                "NDCG@20": LinkPredNDCG(k=20),
                "Precision@20": LinkPredRecall(k=20),
                "Recall@20": LinkPredRecall(k=20),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def common_step(self, batch: DataOrHeteroData) -> [Tensor, Tensor, Adj]:
        if isinstance(batch, HeteroData):
            x_dict = self.recommender(batch)
            pos_scores = self.recommender.predict_link_embedding(
                x_dict, batch[self.target]["pos_edge_label_index"]
            )
            neg_scores = self.recommender.predict_link_embedding(
                x_dict, batch[self.target]["neg_edge_label_index"]
            )
            pos_edge_index = batch[self.target]["pos_edge_label_index"]
            max_src_index = batch[self.target[0]].num_nodes
            max_dst_index = batch[self.target[-1]].num_nodes
        else:
            embed = self.recommender(batch)
            pos_scores = self.recommender.predict_link_embedding(
                embed, batch.pos_edge_label_index
            )
            neg_scores = self.recommender.predict_link_embedding(
                embed, batch.neg_edge_label_index
            )
            pos_edge_index = batch.pos_edge_label_index
            max_src_index = batch.num_nodes
            max_dst_index = batch.num_nodes

        scores = torch.cat((pos_scores, neg_scores), dim=0)
        labels = torch.cat(
            (torch.ones(pos_scores.shape), torch.zeros(neg_scores.shape)), dim=0
        )
        loss = self.recommender.link_pred_loss(scores, labels)

        rec_scores = self.recommender.recommend(
            batch,
            src_index=torch.arange(0, max_src_index),
            dst_index=torch.arange(0, max_dst_index),
            k=20,
        )
        return loss, rec_scores, pos_edge_index

    def training_step(self, batch: DataOrHeteroData, batch_idx: int) -> STEP_OUTPUT:
        loss, recommendations, pos_edge_index = self.common_step(batch)

        metrics = self.train_metrics(recommendations, pos_edge_index)
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log("train/loss", loss, batch_size=1)

        return loss

    def validation_step(self, batch: DataOrHeteroData, batch_idx: int) -> STEP_OUTPUT:
        loss, recommendations, pos_edge_index = self.common_step(batch)

        metrics = self.val_metrics(recommendations, pos_edge_index)
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log("val/loss", loss, batch_size=1)

        return loss

    def test_step(self, batch: DataOrHeteroData, batch_idx: int) -> STEP_OUTPUT:
        loss, recommendations, pos_edge_index = self.common_step(batch)

        metrics = self.test_metrics(recommendations, pos_edge_index)
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
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
