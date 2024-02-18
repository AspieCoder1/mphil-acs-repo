#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, hidden_channels: int = 256, n_heads=8):
        super().__init__()

        self.conv = nn.ModuleList(
            [
                GATConv(
                    64,
                    hidden_channels,
                    heads=n_heads,
                    dropout=0.6,
                    add_self_loops=False,
                ),
                GATConv(
                    hidden_channels * n_heads,
                    hidden_channels,
                    heads=n_heads,
                    dropout=0.6,
                    add_self_loops=False,
                ),
                GATConv(
                    n_heads * hidden_channels,
                    hidden_channels,
                    heads=1,
                    dropout=0.6,
                    add_self_loops=False,
                ),
            ]
        )

    def forward(self, x, edge_index):
        for layer in self.conv:
            x = F.elu(layer(x, edge_index))
        return x
