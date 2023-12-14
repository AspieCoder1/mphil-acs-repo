from typing import Optional

import lightning as L
import torch
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import MovieLens, LastFM


class LinkPredBase(L.LightningDataModule):
    def __init__(self):
        super(LinkPredBase, self).__init__()

    def prepare_data(self) -> None:
        ...


class LastFMDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super(LastFMDataModule, self).__init__()

        self.target = ("user", "to", "artist")
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None

    def prepare_data(self) -> None:
        dataset = LastFM(self.data_dir, transform=T.Constant())

        self.data = dataset[0]
        self.metadata = self.data.metadata()

        print(self.data["user"].num_features)

        self.in_channels = {
            node_type: self.data[node_type].num_features for node_type in
            self.data.node_types
        }

        self.pyg_datamodule = LightningLinkData(
            self.data,
            input_train_edges=torch.concat(
                [
                    self.data[self.target].edge_index,
                    self.data[self.target].train_neg_edge_index
                ], dim=-1
            ),
            input_train_labels=torch.concat(
                [
                    torch.ones(self.data[self.target].edge_index.size(1)),
                    torch.zeros(self.data[self.target].train_neg_edge_index.size(1))
                ], dim=-1
            ),
            input_val_edges=torch.concat(
                [
                    self.data[self.target].val_pos_edge_index,
                    self.data[self.target].val_neg_edge_index,
                ], dim=-1
            ),
            input_val_labels=torch.concat(
                [
                    torch.ones(self.data[self.target].val_pos_edge_index.size(1)),
                    torch.zeros(self.data[self.target].val_neg_edge_index.size(1))
                ], dim=-1
            ),
            input_test_edges=torch.concat(
                [
                    self.data[self.target].test_pos_edge_index,
                    self.data[self.target].test_neg_edge_index,
                ], dim=-1
            ),
            input_test_labels=torch.concat(
                [
                    torch.ones(self.data[self.target].test_pos_edge_index.size(1)),
                    torch.zeros(self.data[self.target].test_neg_edge_index.size(1))
                ], dim=-1
            ),
            loader="full"
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.test_dataloader()


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
