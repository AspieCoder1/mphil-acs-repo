#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data


class GAT(nn.Module):
    def __init__(self, hidden_channels: int = 256, n_heads=8, in_channels: int = 64):
        super().__init__()

        self.conv = Sequential(
            "x, edge_index",
            [
                (
                    GATConv(
                        in_channels,
                        hidden_channels,
                        heads=n_heads,
                        dropout=0.6,
                        add_self_loops=False,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                (
                    GATConv(
                        hidden_channels * n_heads,
                        hidden_channels,
                        heads=n_heads,
                        dropout=0.6,
                        add_self_loops=False,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
                (
                    GATConv(
                        hidden_channels * n_heads,
                        hidden_channels,
                        heads=1,
                        dropout=0.6,
                        add_self_loops=False,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ELU(),
            ],
        )

    def forward(self, data: Data):
        return self.conv(data.x, data.edge_index)
