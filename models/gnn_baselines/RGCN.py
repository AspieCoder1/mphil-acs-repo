#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(
        self, hidden_channels: int, num_nodes: int, num_relations: int, **_kwargs
    ):
        super().__init__()
        self.conv1 = RGCNConv(num_nodes, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)

    def forward(self, data: HeteroData):
        node_type_names = data.node_types
        data = data.to_homogeneous()

        edge_type, edge_index = data.edge_type, data.edge_index
        x = self.conv1(None, edge_index, edge_type)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(x)

        data.update({"x": x})
        data = data.to_heterogeneous(node_type_names=node_type_names)
        return data.x_dict

    def __repr__(self):
        return "RGCN"
