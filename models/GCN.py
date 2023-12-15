import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GCN(nn.Module):
    def __init__(self, hidden_channels: int = 256):
        super().__init__()
        self.conv = nn.ModuleList([
            SAGEConv(-1, hidden_channels, add_self_loops=False),
            SAGEConv(hidden_channels, hidden_channels, add_self_loops=False),
            SAGEConv(hidden_channels, hidden_channels, add_self_loops=False)
        ]
        )

    def forward(self, x, edge_index):
        for layer in self.conv:
            x = F.elu(layer(x, edge_index))
        return x
