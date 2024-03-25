#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch.nn import functional

from models.sheaf_hgnn import utils


class HyperGCN(nn.Module):
    def __init__(self, V, E, X, num_features, num_layers, num_classses, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        h = [d]
        for i in range(l - 1):
            power = l - i + 2
            if args.dname == "citeseer":
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.HyperGCN_fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.HyperGCN_mediators)
        else:
            reapproximate = True
            structure = E
        self.layers = nn.ModuleList(
            [
                utils.HyperGraphConvolution(h[i], h[i + 1], reapproximate, cuda)
                for i in range(l)
            ]
        )
        self.do, self.l = args.dropout, num_layers
        self.structure, self.m = structure, args.HyperGCN_mediators

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        return H
