#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional

from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 256,
        in_channels: Optional[dict[str, int]] = None,
        **_kwargs
    ):
        super().__init__()
        if in_channels is None:
            in_channels = -1

        self.conv = nn.ModuleList(
            [
                HGTConv(
                    in_channels,
                    hidden_channels,
                    heads=8,
                    metadata=metadata,
                ),
                HGTConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    metadata=metadata,
                ),
                HGTConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    metadata=metadata,
                ),
            ]
        )

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        for layer in self.conv:
            x_dict = layer(x_dict, data.edge_index_dict)

        return x_dict

    def __repr__(self):
        return 'HGT'
