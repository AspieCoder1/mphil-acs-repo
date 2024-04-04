#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from models.sheaf_gnn.transductive.disc_models import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
)


class InductiveDiscreteDiagSheafDiffusion(DiscreteDiagSheafDiffusion):

    def __init__(self, _edge_index, args, sheaf_learner):
        super(InductiveDiscreteDiagSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data):
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteDiagSheafDiffusion, self).forward(data)


class InductiveDiscreteBundleSheafDiffusion(DiscreteBundleSheafDiffusion):

    def __init__(self, _edge_index, args, sheaf_learner):
        super(InductiveDiscreteBundleSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data):
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteBundleSheafDiffusion, self).forward(data)


class InductiveDiscreteGeneralSheafDiffusion(DiscreteGeneralSheafDiffusion):

    def __init__(self, _edge_index, args, sheaf_learner):
        super(InductiveDiscreteGeneralSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data):
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteGeneralSheafDiffusion, self).forward(data)
