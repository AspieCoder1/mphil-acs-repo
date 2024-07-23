#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional

import lightning as L
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning import LightningLinkData

from .utils.hgb_datasets import HGBDatasetLP
from .utils.transforms import RemoveSelfLoops, TrainValEdgeSplit

DATA_DIR = "data"


class LinkPredBase(L.LightningDataModule):
    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        num_classes: int = 1,
            hyperparam_tuning: bool = False
    ):
        super(LinkPredBase, self).__init__()
        self.target: tuple[str, str, str] = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None
        self.num_nodes = None
        self.transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
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

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.num_edges = data.num_edges

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.num_nodes
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        self.data = data

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(LastFMDataModule, self).__init__(
            data_dir=f"{data_dir}",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=homogeneous,
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
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(PubMedLPDataModule, self).__init__(
            data_dir=f"{data_dir}",
            target=("DISEASE", "and", "DISEASE"),
            rev_target=("DISEASE", "and", "DISEASE"),
            is_homogeneous=homogeneous,
        )
        self.transform = T.Compose(
            [
                # T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
                TrainValEdgeSplit(target=self.target)
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



# class AmazonBooksDataModule(LinkPredBase):
#     def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
#         super(AmazonBooksDataModule, self).__init__(
#             data_dir=f"{data_dir}/amazon_books",
#             target=("user", "rates", "book"),
#             rev_target=("book", "rated_by", "user"),
#             is_homogeneous=homogeneous,
#         )
#
#     def download_data(self) -> HeteroData:
#         data = AmazonBook(self.data_dir, transform=self.transform)[0]
#         return data
#
#     def __repr__(self):
#         return "AmazonBooks"
#
#
# class MovieLensDatamodule(LinkPredBase):
#     def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
#         super(MovieLensDatamodule, self).__init__(
#             data_dir=f"{data_dir}/movie_lens",
#             target=("user", "rates", "movie"),
#             rev_target=("movie", "rev_rates", "user"),
#             is_homogeneous=homogeneous,
#         )
#
#     def download_data(self) -> HeteroData:
#         data = MovieLens(self.data_dir, transform=self.transform)[0]
#         del data[self.target]["edge_label"]
#         del data[self.rev_target]["edge_label"]
#         # new_edge_labels = torch.ones_like(data[self.target].edge_label)
#         # data[self.target].edge_label = new_edge_labels
#         # data[self.rev_target].edge_label = new_edge_labels
#
#         # print(data[self.target].edge_label)
#         return data
#
#     def __repr__(self):
#         return "MovieLens"
