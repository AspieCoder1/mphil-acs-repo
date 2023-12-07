from typing import Literal, Optional

import lightning as L
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.datasets import HGBDataset


class HGBBaseDataModule(L.LightningDataModule):
    def __init__(self, target: str = "author", num_classes: int = 4,
                 data_dir: str = "data",
                 task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
                 dataset: Literal["IMDB", "DBLP", "ACM", "Freebase"] = "DBLP"
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.num_classes: int = num_classes
        self.task = task
        self.pyg_datamodule: Optional[LightningNodeData] = None
        self.metadata = None
        self.dataset = dataset

    def prepare_data(self) -> None:
        transform = T.Compose(
            [T.Constant(node_types=None), T.RandomNodeSplit()])
        dataset = HGBDataset(root=self.data_dir, name=self.dataset,
                             transform=transform)

        data: HeteroData = dataset[0]
        self.pyg_datamodule = LightningNodeData(
            data,
            input_train_nodes=(self.target, data[self.target].train_mask),
            input_val_nodes=(self.target, data[self.target].val_mask),
            input_test_nodes=(self.target, data[self.target].test_mask),
            loader="full",
            batch_size=128
        )
        self.metadata = data.metadata()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.test_dataloader()


class IMDBDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir=data_dir, task="multilabel", num_classes=5,
                         dataset="IMDB", target="movie")

    def __str__(self):
        return "IMDB"


class DBLPDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__(dataset="DBLP", num_classes=4, target="author",
                         task="multiclass", data_dir=data_dir)

    def __str__(self):
        return "DBLP"


class ACMDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__(
            data_dir=data_dir,
            num_classes=3,
            target="paper",
            dataset="ACM",
            task="multiclass"
        )

    def __str__(self):
        return "ACM"


class FreebaseDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__(
            data_dir=data_dir,
            dataset="Freebase",
            target="book",
            num_classes=5,
            task="multiclass"
        )

    def __str__(self):
        return "Freebase"
