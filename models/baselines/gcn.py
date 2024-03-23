#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, hidden_channels: int = 256, in_channels: int = 64):
        super().__init__()
        self.conv = nn.ModuleList(
            [
                GCNConv(in_channels, hidden_channels, add_self_loops=False),
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
            ]
        )

    def forward(self, x, edge_index):
        for layer in self.conv:
            x = F.elu(layer(x, edge_index))
        return x
