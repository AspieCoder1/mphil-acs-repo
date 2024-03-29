#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch
from torch import nn
from torch.nn import functional
from torch_scatter import scatter_mean

from models.sheaf_hgnn.layers import HypergraphConv


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer
    is implemented in pyg.


    self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs
    """

    def __init__(self, args):
        super(HCHA, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.heads = args.heads
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden // self.heads
        self.init_hedge = args.init_hedge
        self.hyperedge_attr = None

        self.residual = args.residual_HCHA
        #        Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(
            HypergraphConv(
                args.num_features,
                self.MLP_hidden,
                use_attention=args.use_attention,
                heads=self.heads,
            )
        )

        for _ in range(self.num_layers - 2):
            self.convs.append(
                HypergraphConv(
                    self.heads * self.MLP_hidden,
                    self.MLP_hidden,
                    use_attention=args.use_attention,
                    heads=self.heads,
                )
            )
        # Output heads is set to 1 as default
        self.convs.append(
            HypergraphConv(
                self.heads * self.MLP_hidden, args.num_classes, use_attention=False
            )
        )
        if args.cuda in [0, 1]:
            self.device = torch.device(
                "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
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
