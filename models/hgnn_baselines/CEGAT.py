#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, GATConv


class CEGAT(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        heads,
        output_heads,
        dropout,
        normalisation="bn",
    ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if normalisation == "bn":
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(heads * hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(
                GATConv(heads * hid_dim, out_dim, heads=output_heads, concat=False)
            )
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hid_dim * heads, hid_dim))
                self.normalizations.append(nn.Identity())

            self.convs.append(
                GATConv(hid_dim * heads, out_dim, heads=output_heads, concat=False)
            )

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is "Identity"):
                normalization.reset_parameters()

    def forward(self, data):
        #         Assume edge_index is already V2V
        x, edge_index, norm = data.x, data.edge_index, data.norm
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
