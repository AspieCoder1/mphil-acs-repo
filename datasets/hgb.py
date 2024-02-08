from typing import Literal, Optional, Union

import lightning as L
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.datasets import HGBDataset

from .utils import RemoveSelfLoops

DATA_DIR = "data"


class HGBBaseDataModule(L.LightningDataModule):
    def __init__(self, target: str = "author", num_classes: int = 4,
                 data_dir: str = DATA_DIR,
                 task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
                 dataset: Literal["IMDB", "DBLP", "ACM", "Freebase"] = "DBLP",
                 homogeneous: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.num_classes: int = num_classes
        self.task = task
        self.pyg_datamodule: Optional[LightningNodeData] = None
        self.metadata = None
        self.dataset = dataset
        self.num_nodes = None
        self.in_channels: Union[Optional[dict[str, int]], int] = None
        self.homogeneous = homogeneous
        self.edge_index: Optional[Union[dict[str, torch.Tensor], torch.Tensor]] = None
        self.graph_size: Optional[int] = None

    def prepare_data(self) -> None:
        transform = T.Compose(
            [T.Constant(node_types=None), T.RandomNodeSplit(), T.ToUndirected(),
             RemoveSelfLoops(), T.NormalizeFeatures(), T.RemoveDuplicatedEdges()])
        dataset = HGBDataset(root=self.data_dir, name=self.dataset,
                             transform=transform)

        data: Union[HeteroData, Data] = dataset[0].coalesce()
        input_nodes = data[self.target]

        self.edge_index = data.edge_index_dict
        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }
        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes

        if self.homogeneous:
            data = data.to_homogeneous()
            self.edge_index = data.edge_index
            input_nodes = data
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features

        self.pyg_datamodule = LightningNodeData(
            data,
            input_train_nodes=(self.target, input_nodes.train_mask),
            input_val_nodes=(self.target, input_nodes.val_mask),
            input_test_nodes=(self.target, input_nodes.test_mask),
            loader="full",
            batch_size=1
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.test_dataloader()


class IMDBDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR,
                 homogeneous: bool = False):
        super().__init__(
            data_dir=data_dir,
            task="multilabel",
            num_classes=5,
            dataset="IMDB",
            target="movie",
            homogeneous=homogeneous
        )

    def __str__(self):
        return "IMDB"


class DBLPDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR,
                 homogeneous: bool = False):
        super().__init__(
            dataset="DBLP",
            num_classes=4,
            target="author",
            task="multiclass",
            data_dir=data_dir,
            homogeneous=homogeneous
        )

    def __str__(self):
        return "DBLP"


class ACMDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR,
                 homogeneous: bool = False):
        super().__init__(
            data_dir=data_dir,
            dataset="ACM",
            num_classes=3,
            target="paper",
            task="multiclass",
            homogeneous=homogeneous
        )

    def __str__(self):
        return "ACM"
