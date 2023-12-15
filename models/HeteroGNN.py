import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]],
                 hidden_channels: int = 256, out_channels: int = 10,
                 num_layers: int = 3,
                 target: str = "author"):
        super().__init__()
        self.target = target

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(-1, hidden_channels, add_self_loops=False) for
                edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        return x_dict
