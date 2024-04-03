#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import lightning as L
from torch import nn
from torch_geometric.typing import Adj, Tensor, OptTensor, SparseTensor
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, LGConv
from typing import Union, Optional


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
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        alpha_learnable=False,
        conv_layer="LGC",
        **kwargs,
    ):
        super().__init__()
        alpha_string = "alpha" if alpha_learnable else ""
        self.name = f"LGCN_{conv_layer}_{num_layers}_e{embedding_dim}_nodes{num_nodes}_{alpha_string}"
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha_learnable == True:
            alpha_vals = torch.rand(num_layers + 1)
            alpha = nn.Parameter(alpha_vals / torch.sum(alpha_vals))
            print(f"Alpha learnable, initialized to: {alpha.softmax(dim=-1)}")
        else:
            if alpha is None:
                alpha = 1.0 / (num_layers + 1)

            if isinstance(alpha, Tensor):
                assert alpha.size(0) == num_layers + 1
            else:
                alpha = torch.tensor([alpha] * (num_layers + 1))

        self.register_buffer("alpha", alpha)

        self.embedding = nn.Embedding(num_nodes, embedding_dim)

        # initialize convolutional layers
        self.conv_layer = conv_layer
        if conv_layer == "LGC":
            self.convs = nn.ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        elif conv_layer == "GAT":
            # initialize Graph Attention layer with multiple heads
            # initialize linear layers to aggregate heads
            n_heads = 5
            self.convs = nn.ModuleList(
                [
                    GATConv(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        heads=n_heads,
                        dropout=0.5,
                        **kwargs,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.linears = nn.ModuleList(
                [
                    nn.Linear(n_heads * embedding_dim, embedding_dim)
                    for _ in range(num_layers)
                ]
            )

        elif conv_layer == "SAGE":
            #  initialize GraphSAGE conv
            self.convs = nn.ModuleList(
                [
                    SAGEConv(
                        in_channels=embedding_dim, out_channels=embedding_dim, **kwargs
                    )
                    for _ in range(num_layers)
                ]
            )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index: Adj) -> Tensor:
        x = self.embedding.weight

        weights = self.alpha.softmax(dim=-1)
        out = x * weights[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.conv_layer == "GAT":
                x = self.linears[i](x)
            out = out + x * weights[i + 1]

        return out

    def initialize_embeddings(self, data):
        # initialize with the data node features
        self.embedding.weight.data.copy_(data.node_feature)

    def forward(self, edge_index: Adj, edge_label_index: OptTensor = None) -> Tensor:
        if edge_label_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index)

        return self.predict_link_embedding(out, edge_label_index)

    def predict_link(
        self, edge_index: Adj, edge_label_index: OptTensor = None, prob: bool = False
    ) -> Tensor:

        pred = self(edge_index, edge_label_index).sigmoid()
        return pred if prob else pred.round()

    def predict_link_embedding(self, embed: Adj, edge_label_index: Adj) -> Tensor:

        embed_src = embed[edge_label_index[0]]
        embed_dst = embed[edge_label_index[1]]
        return (embed_src * embed_dst).sum(dim=-1)

    def recommend(
        self,
        edge_index: Adj,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
    ) -> Tensor:
        out_src = out_dst = self.get_embedding(edge_index)

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


class RecommenderModule(L.LightningModule):
    def __init__(self):
        super(RecommenderModule, self).__init__()
