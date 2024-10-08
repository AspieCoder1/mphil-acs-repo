#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Literal, Optional, Union

import lightning as L
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.data.lightning import LightningNodeData

from datasets.utils.hgb_datasets import HGBDatasetNC
from datasets.utils.transforms import TrainValNodeSplit, GenerateNodeFeatures, RemoveSelfLoops

DATA_DIR = "data"


class HGBBaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        target: str = "author",
        num_classes: int = 4,
        data_dir: str = DATA_DIR,
        task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
        dataset: Literal["IMDB", "DBLP", "ACM", "Freebase", "PubMed_NC"] = "DBLP",
        homogeneous: bool = False,
        hyperparam_tuning: bool = False,
            feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0',
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
        self.num_node_types: Optional[int] = None
        self.num_edge_types: Optional[int] = None
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None
        self.hyperparam_tuning = hyperparam_tuning
        self.feat_type = feat_type

    def prepare_data(self) -> None:
        transform = T.Compose(
            [
                T.Constant(node_types=None),
                GenerateNodeFeatures(target=self.target, feat_type=self.feat_type),
                TrainValNodeSplit(hyperparam_tuning=self.hyperparam_tuning),
                # T.AddSelfLoops(),
                RemoveSelfLoops(),
                T.RemoveDuplicatedEdges(),
            ]
        )
        dataset = HGBDatasetNC(root=self.data_dir, name=self.dataset,
                               transform=transform)

        data: HeteroData = dataset[0].coalesce()
        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }
        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.edge_index = data.homo_edge_index
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.edge_type_names = data.edge_types
        self.node_type_names = data.node_types
        self.graph_size = data.num_nodes

        self.pyg_datamodule = LightningNodeData(
            data,
            input_train_nodes=(self.target, data[self.target].train_mask),
            input_val_nodes=(self.target, data[self.target].val_mask),
            input_test_nodes=(self.target, data[self.target].test_mask),
            loader="full",
            batch_size=1,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.pyg_datamodule.test_dataloader()


class IMDBDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False,
                 hyperparam_tuning: bool = False,
                 feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0', ):
        super().__init__(
            data_dir=data_dir,
            task="multilabel",
            num_classes=5,
            dataset="IMDB",
            target="movie",
            homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type,
        )

    def __str__(self):
        return "IMDB"


class DBLPDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False,
                 hyperparam_tuning: bool = False,
                 feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0', ):
        super().__init__(
            dataset="DBLP",
            num_classes=4,
            target="author",
            task="multiclass",
            data_dir=data_dir,
            homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )

    def __str__(self):
        return "DBLP"


class ACMDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False,
                 hyperparam_tuning: bool = False,
                 feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0', ):
        super().__init__(
            data_dir=data_dir,
            dataset="ACM",
            num_classes=3,
            target="paper",
            task="multiclass",
            homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )

    def __str__(self):
        return "ACM"


class FreebaseDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False,
                 hyperparam_tuning: bool = False,
                 feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0', ):
        super().__init__(
            data_dir=data_dir,
            dataset="Freebase",
            num_classes=7,
            target="book",
            task="multiclass",
            homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )

    def __str__(self):
        return "Freebase"


class PubMedDataModule(HGBBaseDataModule):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False,
                 hyperparam_tuning: bool = False,
                 feat_type: Literal['feat0', 'feat1', 'feat2'] = 'feat0', ):
        super().__init__(
            dataset="PubMed_NC",
            num_classes=8,
            target="DISEASE",
            task="multiclass",
            data_dir=data_dir,
            homogeneous=homogeneous,
            hyperparam_tuning=hyperparam_tuning,
            feat_type=feat_type
        )

    def __str__(self):
        return "PubMed_NC"


if __name__ == "__main__":
    dm = FreebaseDataModule(data_dir="../data", homogeneous=True)
    dm.prepare_data()
