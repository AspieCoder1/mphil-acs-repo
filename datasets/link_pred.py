from typing import Optional

import lightning as L
import torch
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import MovieLens, LastFM, AmazonBook

from .utils import RemoveSelfLoops


class LinkPredBase(L.LightningDataModule):
    def __init__(self, target: tuple[str, str, str], data_dir: str = "data"):
        super(LinkPredBase, self).__init__()
        self.target = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None
        self.num_nodes = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(self, data_dir: str = "data"):
        super(LastFMDataModule, self).__init__(data_dir=f"{data_dir}/lastfm",
                                               target=("user", "to", "artist"))

    def prepare_data(self) -> None:
        transform = T.Compose([
            T.Constant(),
            T.ToUndirected(),
            T.NormalizeFeatures(),
            RemoveSelfLoops(),
        ]
        )

        data = LastFM(self.data_dir, transform=transform)[0]

        del data['user', 'artist']['train_neg_edge_index']
        del data['user', 'artist']['val_pos_edge_index']
        del data['user', 'artist']['val_neg_edge_index']
        del data['user', 'artist']['test_pos_edge_index']
        del data['user', 'artist']['test_neg_edge_index']

        split = T.RandomLinkSplit(edge_types=("user", "to", "artist"),
                                  is_undirected=True,
                                  rev_edge_types=("artist", "to", "user"))

        self.train_data, self.val_data, self.test_data = split(data)

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in
            data.node_types
        }


class AmazonBooksDataModule(LinkPredBase):
    def __init__(self, data_dir: str = "data"):
        super(AmazonBooksDataModule, self).__init__(data_dir=f"{data_dir}/amazon_books",
                                                    target=("user", "rates", "book"))

    def prepare_data(self) -> None:
        transform = T.Compose([
            T.Constant(),
            T.ToUndirected(),
            T.NormalizeFeatures(),
            RemoveSelfLoops(),
        ]
        )

        split = T.RandomLinkSplit(edge_types=('user', 'rates', 'book'),
                                  is_undirected=True,
                                  rev_edge_types=('book', 'rated_by', 'user'))

        dataset = AmazonBook(self.data_dir, transform=transform)[0]

        self.train_data, self.val_data, self.test_data = split(dataset)

        self.metadata = dataset.metadata()
        self.num_nodes = dataset.num_nodes

        self.in_channels = {
            node_type: dataset[node_type].num_features for node_type in
            dataset.node_types
        }


class MovieLensDataset(L.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super(MovieLensDataset, self).__init__()
        self.data_dir = data_dir
        self.pyg_datamodule: Optional[LightningLinkData] = None
        self.edge_type = ("user", "rates", "movie")
        self.metadata = None
        self.train_split = None
        self.valid_split = None
        self.test_split = None
        self.task = "multiclass"
        self.num_classes = 7

    def prepare_data(self) -> None:
        transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.RandomLinkSplit(
                    edge_types=self.edge_type,
                    rev_edge_types=("movie", "rev_rates", "user"),
                )
            ]
        )
        dataset = MovieLens(self.data_dir, transform=transform)

        train_split, valid_split, test_split = dataset[0]
        weights = torch.bincount(train_split[self.edge_type].edge_label,
                                 minlength=self.num_classes)
        self.weights = weights / weights.sum()
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split

        self.metadata = train_split.metadata()
        self.pyg_datamodule = LightningLinkData(
            train_split,
            loader="full"
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(
            self.train_split,
            loader="full"
        ).full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(
            self.valid_split,
            loader="full"
        ).full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(
            self.test_split,
            loader="full"
        ).full_dataloader()
