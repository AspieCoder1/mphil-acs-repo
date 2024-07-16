#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from models.sheaf_gnn.config import SheafLearners
from models.sheaf_gnn.sheaf_models import (
    TypeConcatSheafLearner,
    LocalConcatSheafLearner,
    TypeEnsembleSheafLearner,
    NodeTypeConcatSheafLearner,
    NodeTypeSheafLearner,
    EdgeTypeSheafLearner,
    EdgeTypeConcatSheafLearner,
    TypeSheafLearner,
)


def init_sheaf_learner(sheaf_type):
    if sheaf_type == SheafLearners.type_concat:
        sheaf_learner = TypeConcatSheafLearner
    elif sheaf_type == SheafLearners.local_concat:
        sheaf_learner = LocalConcatSheafLearner
    elif sheaf_type == SheafLearners.type_ensemble:
        sheaf_learner = TypeEnsembleSheafLearner
    elif sheaf_type == SheafLearners.node_type_concat:
        sheaf_learner = NodeTypeConcatSheafLearner
    elif sheaf_type == SheafLearners.node_type:
        sheaf_learner = NodeTypeSheafLearner
    elif sheaf_type == SheafLearners.edge_type:
        sheaf_learner = EdgeTypeSheafLearner
    elif sheaf_type == SheafLearners.edge_type_concat:
        sheaf_learner = EdgeTypeConcatSheafLearner
    else:
        sheaf_learner = TypeSheafLearner
    return sheaf_learner
