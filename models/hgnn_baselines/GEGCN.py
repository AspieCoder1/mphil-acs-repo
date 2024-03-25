#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch.nn import functional
from torch_geometric.nn import MessagePassing, GCNConv


class CEGCN(MessagePassing):
    def __init__(
        self, in_dim, hid_dim, out_dim, num_layers, dropout, Normalization="bn"
    ):
        super(CEGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == "bn":
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
        else:  # default no normalizations
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.Identity())

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))

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
            x = conv(x, edge_index, norm)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, norm)
        return x
