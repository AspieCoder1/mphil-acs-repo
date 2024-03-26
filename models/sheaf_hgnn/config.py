#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from enum import auto
from core.trainer import TrainerArgs

from strenum import PascalCaseStrEnum, LowercaseStrEnum, SnakeCaseStrEnum


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


class SheafModelTypes(PascalCaseStrEnum):
    SheafHyperGNN = "SheafHyperGNN"
    SheafHyperGCN = "SheafHyperGCN"


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


@dataclass
class Config:
    model: SheafModelTypes
    restriction_maps: HGNNSheafTypes
    trainer: TrainerArgs
