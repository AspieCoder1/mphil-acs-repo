#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional, Union

import lightning as L
import torch
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import MovieLens, LastFM, AmazonBook

from .utils import RemoveSelfLoops

DATA_DIR = "data"


class LinkPredBase(L.LightningDataModule):
    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        num_classes: int = 1,
    ):
        super(LinkPredBase, self).__init__()
        self.target: tuple[str, str, str] = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None
        self.num_nodes = None
        self.train_data: Optional[Union[Data, HeteroData]] = None
        self.val_data: Optional[Union[Data, HeteroData]] = None
        self.test_data: Optional[Union[Data, HeteroData]] = None
        self.transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
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

    def download_data(self) -> HeteroData: ...

    def prepare_data(self) -> None:
        data = self.download_data()

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types

        split = T.RandomLinkSplit(
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            # split_labels=True,
            add_negative_train_samples=False,
            # neg_sampling_ratio=0.6,
            rev_edge_types=self.rev_target,
        )

        self.train_data, self.val_data, self.test_data = split(data)

        # if self.is_homogeneous:
        #     self.train_data = self.train_data.to_homogeneous()
        #     self.val_data = self.val_data.to_homogeneous()
        #     self.test_data = self.test_data.to_homogeneous()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, is_homogeneous: bool = False):
        super(LastFMDataModule, self).__init__(
            data_dir=f"{data_dir}/lastfm",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=is_homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = LastFM(self.data_dir, transform=self.transform)[0]

        del data[self.target]["train_neg_edge_index"]
        del data[self.target]["val_pos_edge_index"]
        del data[self.target]["val_neg_edge_index"]
        del data[self.target]["test_pos_edge_index"]
        del data[self.target]["test_neg_edge_index"]

        return data


class AmazonBooksDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, is_homogeneous: bool = False):
        super(AmazonBooksDataModule, self).__init__(
            data_dir=f"{data_dir}/amazon_books",
            target=("user", "rates", "book"),
            rev_target=("book", "rated_by", "user"),
            is_homogeneous=is_homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = AmazonBook(self.data_dir, transform=self.transform)[0]
        return data


class MovieLensDatamodule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, is_homogeneous: bool = False):
        super(MovieLensDatamodule, self).__init__(
            data_dir=f"{data_dir}/movie_lens",
            target=("user", "rates", "movie"),
            rev_target=("movie", "rated_by", "user"),
            is_homogeneous=is_homogeneous,
        )
        # self.data_dir = data_dir
        # self.pyg_datamodule: Optional[LightningLinkData] = None
        # self.edge_type = ("user", "rates", "movie")
        # self.metadata = None
        # self.train_split = None
        # self.valid_split = None
        # self.test_split = None

    def download_data(self) -> HeteroData:
        data = MovieLens(self.data_dir, transform=self.transform)[0]
        # del data[self.edge_type].edge_label
        return data

    # def prepare_data(self) -> None:
    #     transform = T.Compose(
    #         [
    #             T.Constant(),
    #             T.ToUndirected(),
    #             T.RandomLinkSplit(
    #                 edge_types=self.edge_type,
    #                 split_labels=True,
    #                 neg_sampling_ratio=0.6,
    #                 rev_edge_types=("movie", "rev_rates", "user"),
    #             ),
    #         ]
    #     )
    #     data = MovieLens(self.data_dir)[0]
    #     del data[self.edge_type].edge_label
    #     data = transform(data)
    #
    #     train_split, valid_split, test_split = data
    #     self.train_split = train_split
    #     self.valid_split = valid_split
    #     self.test_split = test_split
    #
    #     self.metadata = train_split.metadata()
    #     self.pyg_datamodule = LightningLinkData(train_split, loader="full")
