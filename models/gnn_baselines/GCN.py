#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, hidden_channels: int = 256, in_channels: int = 64):
        super().__init__()

        self.conv = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(in_channels, hidden_channels, add_self_loops=False),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                (
                    GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                (
                    GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
            ],
        )

    def forward(self, data: Data):
        return self.conv(data.x, data.edge_index)
