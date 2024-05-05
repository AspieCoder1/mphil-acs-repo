#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Literal, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F, Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_add


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer
    is implemented in pyg.


    self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_attention: bool = True,
        heads: int = 1,
        residual_connections: bool = False,
        symdegnorm: bool = True,
        init_hedge: Literal["rand", "avg"] = "rand",
        cuda: int = 0,
        **_kwargs
    ):
        super(HCHA, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout  # Note that default is 0.6
        self.symdegnorm = symdegnorm
        self.heads = heads
        self.num_features = in_channels
        self.MLP_hidden = hidden_channels // self.heads
        self.init_hedge = init_hedge
        self.hyperedge_attr = None
        self.use_attention = use_attention

        self.residual = residual_connections
        #        Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(
            HypergraphConv(
                in_channels,
                self.MLP_hidden,
                use_attention=use_attention,
                heads=self.heads,
            )
        )

        for _ in range(self.num_layers - 2):
            self.convs.append(
                HypergraphConv(
                    self.heads * self.MLP_hidden,
                    self.MLP_hidden,
                    use_attention=use_attention,
                    heads=self.heads,
                )
            )
        # Output heads is set to 1 as default
        self.convs.append(
            HypergraphConv(
                self.heads * self.MLP_hidden, out_channels, use_attention=False, heads=1
            )
        )
        if cuda in [0, 1]:
            self.device = torch.device(
                "cuda:" + str(cuda) if torch.cuda.is_available() else "cpu"
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        if type == "rand":
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == "avg":
            hyperedge_attr = scatter_mean(
                x[hyperedge_index[0]], hyperedge_index[1], dim=0
            )
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]  # data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        # hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        if self.hyperedge_attr is None:
            if hasattr(data, "hyperedge_attr"):
                self.hyperedge_attr = data.hyperedge_attr
            else:
                self.hyperedge_attr = self.init_hyperedge_attr(
                    type=self.init_hedge,
                    num_edges=num_edges,
                    x=x,
                    hyperedge_index=edge_index,
                )

        for i, conv in enumerate(self.convs[:-1]):
            # print(i)
            x = F.elu(conv(x, edge_index, hyperedge_attr=self.hyperedge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        #         x = F.dropout(x, p=self.dropout, training=self.training)

        # print("Ok")
        x = self.convs[-1](x, edge_index)

        return x

    def __repr__(self):
        if self.use_attention:
            return 'HCHA'
        else:
            return 'HGNN'


class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=False,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        residual=False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.residual = residual
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False)
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None,
        hyperedge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        num_nodes = hyperedge_index[0].max().item() + 1

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)
        data_x = x
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # pdb.set_trace()
        D = scatter_add(
            hyperedge_weight[hyperedge_index[1]],
            hyperedge_index[0],
            dim=0,
            dim_size=num_nodes,
        )
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges,
        )
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(
            hyperedge_index, x=x, norm=B, alpha=alpha, size=(num_nodes, num_edges)
        )
        out = self.propagate(
            hyperedge_index.flip([0]),
            x=out,
            norm=D,
            alpha=alpha,
            size=(num_edges, num_nodes),
        )

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
