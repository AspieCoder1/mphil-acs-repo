from enum import auto
from typing import Union

from strenum import UppercaseStrEnum
from torch_geometric.nn import to_hetero_with_bases

from datasets.hgb import HGBBaseDataModule
from datasets.link_pred import LinkPredBase
from models.baselines import HAN, HGT, HeteroGNN, RGCN, GCN, GAT


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
