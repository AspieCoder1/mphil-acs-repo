#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from models.sheaf_gnn.transductive.disc_models import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
)


class InductiveDiscreteDiagSheafDiffusion(DiscreteDiagSheafDiffusion):

    def __init__(self, _edge_index, args):
        super(InductiveDiscreteDiagSheafDiffusion, self).__init__(
            edge_index=None, args=args
        )

    def forward(self, x, edge_index):
        self.update_edge_index(edge_index)
        return super(InductiveDiscreteDiagSheafDiffusion, self).forward(x)


class InductiveDiscreteBundleSheafDiffusion(DiscreteBundleSheafDiffusion):

    def __init__(self, _edge_index, args):
        super(InductiveDiscreteBundleSheafDiffusion, self).__init__(
            edge_index=None, args=args
        )

    def forward(self, x, edge_index):
        self.update_edge_index(edge_index)
        return super(InductiveDiscreteBundleSheafDiffusion, self).forward(x)


class InductiveDiscreteGeneralSheafDiffusion(DiscreteGeneralSheafDiffusion):

    def __init__(self, _edge_index, args):
        super(InductiveDiscreteGeneralSheafDiffusion, self).__init__(
            edge_index=None, args=args
        )

    def forward(self, x, edge_index):
        self.update_edge_index(edge_index)
        return super(InductiveDiscreteGeneralSheafDiffusion, self).forward(x)
