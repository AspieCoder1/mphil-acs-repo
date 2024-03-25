#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from enum import auto

from strenum import PascalCaseStrEnum, LowercaseStrEnum, SnakeCaseStrEnum

args_dict = {
    "num_features": 10,  # number of node features
    "num_classes": 4,  # number of classes
    "All_num_layers": 2,  # number of layers
    "dropout": 0.3,  # dropout rate
    "MLP_hidden": 256,  # dimension of hidden state (for most of the layers)
    "AllSet_input_norm": True,  # normalising the input at each layer
    "residual_HCHA": False,  # using or not a residual connectoon per sheaf layer
    "heads": 6,  # dimension of reduction map (d)
    "init_hedge": "avg",  # how to compute hedge features when needed. options: 'avg'or 'rand'
    "sheaf_normtype": "sym_degree_norm",  # the type of normalisation for the sheaf Laplacian. options: 'degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm'
    "sheaf_act": "tanh",  # non-linear activation used on tpop of the d x d restriction maps. options: 'sigmoid', 'tanh', 'none'
    "sheaf_left_proj": False,  # multiply to the left with IxW or not
    "dynamic_sheaf": False,  # infering a differetn sheaf at each layer or use ta shared one
    "sheaf_pred_block": "cp_decomp",  # indicated the type of model used to predict the restriction maps. options: 'MLP_var1', 'MLP_var3' or 'cp_decomp'
    "sheaf_dropout": False,  # use dropout in the sheaf layer or not
    "sheaf_special_head": False,  # if True, add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
    "rank": 2,  # only for LowRank type of sheaf. mention the rank of the reduction matrix
    "HyperGCN_mediators": True,  # only for the Non-Linear sheaf laplacian. Indicates if mediators are used when computing the non-linear Laplacian (same as in HyperGCN)
    "cuda": 0,
}


class HGNNSheafTypes(PascalCaseStrEnum):
    DiagSheafs = auto()
    GeneralSheafs = auto()
    OrthoSheafs = auto()
    LowRankSheafs = auto()


class SheafActivations(LowercaseStrEnum):
    sigmoid = auto()
    none = auto()
    tanh = auto()


class SheafNormTypes(SnakeCaseStrEnum):
    degree_norm = auto()
    block_norm = auto()
    sym_degree_norm = auto()
    sym_block_norm = auto()


class SheafPredictionBlockTypes(SnakeCaseStrEnum):
    MLP_var1 = "MLP_var1"
    MLP_var3 = "MLP_var3"
    cp_decomp = auto()


@dataclass
class SheafHGNNConfig:
    num_features: int
    num_classes: int
    All_num_layers: int
    dropout: float
    MLP_hidden: int
    AllSet_input_norm: bool
    residual_HCHA: bool
    heads: int
    init_hedge: str
    sheaf_normtype: SheafNormTypes
    sheaf_act: SheafActivations
    sheaf_left_proj: bool
    dynamic_sheaf: bool
    sheaf_pred_block: SheafPredictionBlockTypes
    sheaf_dropout: float
    sheaf_special_head: bool
    rank: int
    HyperGCN_mediators: bool
    cuda: int
