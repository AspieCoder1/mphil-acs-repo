#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from enum import auto
from typing import Union, Type

from strenum import UppercaseStrEnum
from torch_geometric.nn import to_hetero_with_bases

from core.sheaf_configs import ModelTypes
from datasets.hgb import HGBBaseDataModule
from datasets.link_pred import LinkPredBase
from models.baselines import HAN, HGT, HeteroGNN, RGCN, GCN, GAT
from models.sheaf_gnn import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
    DiagSheafDiffusion,
    BundleSheafDiffusion,
    GeneralSheafDiffusion,
)
from models.sheaf_gnn.inductive import (
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion,
)
from models.sheaf_gnn.sheaf_base import (
    SheafDiffusion,
    SheafDiffusion as SheafDiffusionInductive,
)


class Models(UppercaseStrEnum):
    HAN = auto()
    HGT = auto()
    HGCN = auto()
    RGCN = auto()
    GCN = auto()
    GAT = auto()


def get_model(model: Models, datamodule: Union[HGBBaseDataModule, LinkPredBase]):
    if model == Models.HAN:
        return HAN(
            datamodule.metadata,
            in_channels=datamodule.in_channels,
            hidden_channels=256
        ), False
    elif model == Models.HGT:
        return HGT(
            datamodule.metadata,
            in_channels=datamodule.in_channels,
            hidden_channels=256
        ), False
    elif model == Models.HGCN:
        return HeteroGNN(
            datamodule.metadata,
            in_channels=datamodule.in_channels,
            hidden_channels=256,
            target=datamodule.target,
            num_layers=3
        ), False
    elif model == Models.RGCN:
        return RGCN(
            hidden_channels=256,
            num_nodes=datamodule.num_nodes,
            num_relations=len(datamodule.metadata[1]),
        ), False
    elif model == Models.GCN:
        gcn = GCN(
            hidden_channels=256
        )

        return to_hetero_with_bases(gcn, datamodule.metadata, num_bases=3,
                                    in_channels={'x': 64}), True
    else:
        gat = GAT(
            hidden_channels=256
        )
        return to_hetero_with_bases(gat, datamodule.metadata, num_bases=3,
                                    in_channels={'x': 64}), True


def get_sheaf_model(model: ModelTypes) -> Type[SheafDiffusion]:
    if model == ModelTypes.DiagSheaf:
        return DiscreteDiagSheafDiffusion
    if model == ModelTypes.BundleSheaf:
        return DiscreteBundleSheafDiffusion
    if model == ModelTypes.GeneralSheaf:
        return DiscreteGeneralSheafDiffusion
    if model == ModelTypes.DiagSheafODE:
        return DiagSheafDiffusion
    if model == ModelTypes.BundleSheafODE:
        return BundleSheafDiffusion
    if model == ModelTypes.GeneralSheafODE:
        return GeneralSheafDiffusion


def get_inductive_sheaf_model(model: ModelTypes) -> Type[SheafDiffusionInductive]:
    if model == ModelTypes.DiagSheaf:
        return InductiveDiscreteDiagSheafDiffusion
    if model == ModelTypes.BundleSheaf:
        return InductiveDiscreteBundleSheafDiffusion
    return InductiveDiscreteGeneralSheafDiffusion
