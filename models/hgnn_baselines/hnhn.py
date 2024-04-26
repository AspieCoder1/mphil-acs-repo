#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import torch
from torch import nn
from torch.nn import functional as F, Linear
from torch_geometric.nn import MessagePassing


class HNHN(nn.Module):
    """ """

    def __init__(
        self,
        num_layers: int,
        num_features: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        nonlinear_inbetween: bool = True,
    ):
        super(HNHN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(
                HNHNConv(
                    num_features,
                    hidden_channels,
                    out_channels,
                    nonlinear_inbetween=nonlinear_inbetween,
                )
            )
        else:
            self.convs.append(
                HNHNConv(
                    num_features,
                    hidden_channels,
                    hidden_channels,
                    nonlinear_inbetween=nonlinear_inbetween,
                )
            )
            for _ in range(self.num_layers - 2):
                self.convs.append(
                    HNHNConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        nonlinear_inbetween=nonlinear_inbetween,
                    )
                )
            self.convs.append(
                HNHNConv(
                    hidden_channels,
                    hidden_channels,
                    out_channels,
                    nonlinear_inbetween=nonlinear_inbetween,
                )
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x

        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x


class HNHNConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads=1,
        nonlinear_inbetween=True,
        concat=True,
        bias=True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween

        # preserve variable heads for later use (attention)
        self.heads = heads
        self.concat = True
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()
        # glorot(self.weight_v2e)
        # glorot(self.weight_e2v)
        # zeros(self.bias)

    def forward(self, x, data):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        hyperedge_index = data.edge_index
        hyperedge_weight = None
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.weight_v2e(x)

        #         ipdb.set_trace()
        #         x = torch.matmul(torch.diag(data.D_v_beta), x)
        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = "source_to_target"
        out = self.propagate(
            hyperedge_index, x=x, norm=data.D_e_beta_inv, size=(num_nodes, num_edges)
        )

        if self.nonlinear_inbetween:
            out = F.relu(out)

        # sanity check
        out = torch.squeeze(out, dim=1)

        out = self.weight_e2v(out)

        #         out = torch.matmul(torch.diag(data.D_e_alpha), out)
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = "target_to_source"
        out = self.propagate(
            hyperedge_index, x=out, norm=data.D_v_alpha_inv, size=(num_edges, num_nodes)
        )

        return out

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
