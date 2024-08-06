#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import logging
from typing import Optional, Literal

import lightning as L
import torch
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.data.lightning import LightningLinkData

from .utils.hgb_datasets import HGBDatasetLP
from .utils.transforms import TrainValEdgeSplit, GenerateNodeFeatures

DATA_DIR = "data"

logger = logging.getLogger(__name__)


class LinkPredBase(L.LightningDataModule):
    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        num_classes: int = 1,
            hyperparam_tuning: bool = False,
            feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat2',
    ):
        super(LinkPredBase, self).__init__()
        self.target: tuple[str, str, str] = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.edge_index=None
        self.in_channels = None
        self.num_nodes = None
        self.transform = T.Compose(
            [
                T.Constant(),
                GenerateNodeFeatures(target=self.target, feat_type=feat_type),
                T.AddSelfLoops(),
                TrainValEdgeSplit(target=self.target,
                                  hyperparam_tuning=hyperparam_tuning)
            ]
        )
        self.rev_target = rev_target
        self.batch_size = 1
        self.is_homogeneous = is_homogeneous
        self.num_classes = num_classes
        self.graph_size = 0
        self.task = "binary"
        self.num_node_types: int = 0
        self.num_edge_types: int = 0
        self.num_edges: int = 0
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None

    def download_data(self) -> HeteroData: ...

    def prepare_data(self) -> None:
        data = self.download_data()

        data: HeteroData = data.coalesce()
        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }
        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.edge_index, node_slices, edge_slices = to_homogeneous_edge_index(data)
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.edge_type_names = data.edge_types
        self.node_type_names = data.node_types
        self.graph_size = data.num_nodes

        sizes = [offset[1] - offset[0] for offset in node_slices.values()]
        sizes = torch.tensor(sizes, dtype=torch.long)
        node_type = torch.arange(len(sizes))
        data.node_type = node_type.repeat_interleave(sizes)

        sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
        sizes = torch.tensor(sizes, dtype=torch.long)
        edge_type = torch.arange(len(sizes))
        data.edge_type = edge_type.repeat_interleave(sizes)

        self.data = data

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(
            self,
            data_dir: str = DATA_DIR,
            homogeneous: bool = False,
            hyperparam_tuning: bool = False,
            feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat2'
    ):
        super(LastFMDataModule, self).__init__(
            data_dir=f"{data_dir}",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )

    def download_data(self) -> HeteroData:
        data = HGBDatasetLP(
            root=self.data_dir,
            name='lastfm',
            transform=self.transform
        )[0]
        return data

    def __repr__(self):
        return "LastFM"


class PubMedLPDataModule(LinkPredBase):
    def __init__(
            self,
            data_dir: str = DATA_DIR,
            homogeneous: bool = False,
            hyperparam_tuning: bool = False,
            feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat2'
    ):
        super(PubMedLPDataModule, self).__init__(
            data_dir=f"{data_dir}",
            target=("DISEASE", "and", "DISEASE"),
            rev_target=("DISEASE", "and", "DISEASE"),
            is_homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )
        self.transform = T.Compose(
            [
                GenerateNodeFeatures(target=self.target, feat_type=feat_type),
                T.AddSelfLoops(),
                TrainValEdgeSplit(target=self.target,
                                  hyperparam_tuning=hyperparam_tuning),
            ]
        )

    def download_data(self) -> HeteroData:
        data = HGBDatasetLP(
            root=self.data_dir,
            name='pubmed_lp',
            transform=self.transform
        )[0]
        return data

    def __repr__(self):
        return "PubMed_LP"