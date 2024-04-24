#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import NamedTuple, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor


class HAN(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 256,
        in_channels: Optional[dict[str, int]] = None,
        **_kwargs
    ):
        super().__init__()

        if isinstance(in_channels, DictConfig):
            in_channels = dict(in_channels)

        if in_channels is None:
            in_channels = -1

        self.conv = nn.ModuleList(
            [
                HANConv(
                    in_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                ),
                HANConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
                    metadata=metadata,
                ),
                HANConv(
                    hidden_channels,
                    hidden_channels,
                    heads=8,
                    dropout=0.6,
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
        return "HAN"
