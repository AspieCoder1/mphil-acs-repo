#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int],
        hidden_channels: int = 256,
        num_layers: int = 3,
        target: str = "author",
        **_kwargs
    ):

        super().__init__()
        self.target = target

        self.convs = nn.ModuleList()
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(
                        in_channels=(
                            in_channels[edge_type[0]],
                            in_channels[edge_type[-1]],
                        ),
                        out_channels=hidden_channels,
                        add_self_loops=False,
                    )
                    for edge_type in metadata[1]
                }
            )
        )
        for i in range(num_layers - 1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        add_self_loops=False,
                    )
                    for edge_type in metadata[1]
                }
            )
            self.convs.append(conv)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        return x_dict

    def __repr__(self):
        return 'HGCN'
