#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import torch
from torch import nn, nn as nn
from torch.nn import functional as F, Linear


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNNConv(in_ch, n_hid)
        self.hgc2 = HGNNConv(n_hid, n_class)

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        x = data.x
        G = data.edge_index

        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNNConv, self).__init__()

        self.lin = Linear(in_ft, out_ft, bias=bias)

    #         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
    #         if bias:
    #             self.bias = Parameter(torch.Tensor(out_ft))
    #         else:
    #             self.register_parameter('bias', None)
    #         self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    #         stdv = 1. / math.sqrt(self.weight.size(1))
    #         self.weight.data.uniform_(-stdv, stdv)
    #         if self.bias is not None:
    #             self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        #         x = data.x
        #         G = data.edge_index

        x = self.lin(x)
        #         x = x.matmul(self.weight)
        #         if self.bias is not None:
        #             x = x + self.bias
        x = torch.matmul(G, x)
        return x
