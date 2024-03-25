#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from torch import nn
from torch.nn import functional

from models.sheaf_hgnn.layers import HNHNConv


class HNHN(nn.Module):
    """ """

    def __init__(self, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout

        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(
                HNHNConv(
                    args.num_features,
                    args.MLP_hidden,
                    args.num_classes,
                    nonlinear_inbetween=args.HNHN_nonlinear_inbetween,
                )
            )
        else:
            self.convs.append(
                HNHNConv(
                    args.num_features,
                    args.MLP_hidden,
                    args.MLP_hidden,
                    nonlinear_inbetween=args.HNHN_nonlinear_inbetween,
                )
            )
            for _ in range(self.num_layers - 2):
                self.convs.append(
                    HNHNConv(
                        args.MLP_hidden,
                        args.MLP_hidden,
                        args.MLP_hidden,
                        nonlinear_inbetween=args.HNHN_nonlinear_inbetween,
                    )
                )
            self.convs.append(
                HNHNConv(
                    args.MLP_hidden,
                    args.MLP_hidden,
                    args.num_classes,
                    nonlinear_inbetween=args.HNHN_nonlinear_inbetween,
                )
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x

        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x
