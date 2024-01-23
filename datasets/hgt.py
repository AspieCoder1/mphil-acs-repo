from typing import Literal, Optional

import lightning as L
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData
from torch_geometric.datasets import HGBDataset
from torch_geometric.loader import HGTLoader

DATA = "data"


class HGTBaseDataModule(L.LightningDataModule):
    def __init__(self, target: str = "author", num_classes: int = 4,
                 data_dir: str = DATA,
                 task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
                 dataset: Literal["IMDB", "DBLP", "ACM", "Freebase"] = "DBLP"
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.num_classes: int = num_classes
        self.task = task
        self.data: Optional[HeteroData] = None
        self.metadata = None
        self.dataset = dataset
        self.in_channels: Optional[dict[str, int]] = None

    def prepare_data(self) -> None:
        transform = T.Compose(
            [T.Constant(node_types=None), T.RandomNodeSplit()])
        dataset = HGBDataset(root=self.data_dir, name=self.dataset,
                             transform=transform)

        self.data: HeteroData = dataset[0]
        self.in_channels = {
            node_type: self.data[node_type].num_features for node_type in
            self.data.node_types
        }
        self.metadata = self.data.metadata()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return HGTLoader(
            self.data,
            num_samples={key: [512] * 4 for key in self.data.node_types},
            batch_size=128,
            input_nodes=(self.target, self.data[self.target].train_mask),
            num_workers=7,
            persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return HGTLoader(
            self.data,
            num_samples={key: [512] * 4 for key in self.data.node_types},
            batch_size=128,
            input_nodes=(self.target, self.data[self.target].val_mask),
            num_workers=7,
            persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return HGTLoader(
            self.data,
            num_samples={key: [512] * 10 for key in self.data.node_types},
            batch_size=128,
            input_nodes=(self.target, self.data[self.target].test_mask),
            num_workers=7,
            persistent_workers=True
        )


class HGTIMDBDataModule(HGTBaseDataModule):
    def __init__(self, data_dir: str = DATA):
        super().__init__(data_dir=data_dir, task="multilabel", num_classes=5,
                         dataset="IMDB", target="movie")

    def __str__(self):
        return "IMDB"


class HGTDBLPDataModule(HGTBaseDataModule):
    def __init__(self, data_dir: str = DATA):
        super().__init__(dataset="DBLP", num_classes=4, target="author",
                         task="multiclass", data_dir=data_dir)

    def __str__(self):
        return "DBLP"


class HGTACMDataModule(HGTBaseDataModule):
    def __init__(self, data_dir: str = DATA):
        super().__init__(
            data_dir=data_dir,
            num_classes=3,
            target="paper",
            dataset="ACM",
            task="multiclass"
        )

    def __str__(self):
        return "ACM"


class HGTFreebaseDataModule(HGTBaseDataModule):
    def __init__(self, data_dir: str = DATA):
        super().__init__(
            data_dir=data_dir,
            dataset="Freebase",
            target="book",
            num_classes=5,
            task="multiclass"
        )

    def __str__(self):
        return "Freebase"
