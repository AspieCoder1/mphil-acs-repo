#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from torch_geometric.typing import Adj, Tensor, OptTensor


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


class Recommender(torch.nn.Module):
    """
    Here we adapt the LightGCN model from Torch Geometric for our purposes. We allow
    for customizable convolutional layers, custom embeddings. In addition, we deifne some
    additional custom functions.

    """

    def __init__(
        self,
        encoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder

    def get_embedding(self, data: Data) -> Tensor:
        x = self.embedding.weight
        return self.encoder(data.x, data.edge_index)

    def forward(self, data: Data) -> Tensor:
        out = self.get_embedding(data.edge_index)

        return self.predict_link_embedding(out, data.edge_label_index)

    def predict_link(self, data: Data, prob: bool = False) -> Tensor:

        pred = self(data).sigmoid()
        return pred if prob else pred.round()

    def predict_link_embedding(self, embed: Adj, edge_label_index: Adj) -> Tensor:

        embed_src = embed[edge_label_index[0]]
        embed_dst = embed[edge_label_index[1]]
        return (embed_src * embed_dst).sum(dim=-1)

    def recommend(
        self,
        data: Data,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
    ) -> Tensor:
        out_src = out_dst = self.get_embedding(data)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
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
        return loss_fn(pos_edge_rank, neg_edge_rank, self.embedding.weight)

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.num_nodes}, "
            f"{self.embedding_dim}, num_layers={self.num_layers})"
        )
