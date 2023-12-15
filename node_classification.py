from dataclasses import dataclass
from enum import auto

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from strenum import UppercaseStrEnum
from torch_geometric.nn import to_hetero, to_hetero_with_bases

from datasets.hgb import (
    DBLPDataModule,
    ACMDataModule,
    IMDBDataModule,
    HGBBaseDataModule
)
from datasets.hgt import (
    HGTDBLPDataModule,
    HGTACMDataModule,
    HGTIMDBDataModule,
    HGTBaseDataModule
)
from models import (
    HAN,
    GCN,
    HGT,
    HeteroGNN,
    RGCN,
    GAT
)
from models.NodeClassifier import NodeClassifier


class Datasets(UppercaseStrEnum):
    DBLP = auto()
    ACM = auto()
    IMDB = auto()


class Models(UppercaseStrEnum):
    HAN = auto()
    HGT = auto()
    HGCN = auto()
    RGCN = auto()
    GCN = auto()
    GAT = auto()


@dataclass
class Config:
    dataset: Datasets
    model: Models
    patience: int = 100


cs = ConfigStore.instance()
cs.store("config", Config)


def get_dataset(dataset: Datasets) -> HGBBaseDataModule:
    if dataset == Datasets.DBLP:
        return DBLPDataModule()
    elif dataset == Datasets.ACM:
        return ACMDataModule()
    else:
        return IMDBDataModule()


def get_dataset_hgt(dataset: Datasets) -> HGTBaseDataModule:
    if dataset == Datasets.DBLP:
        return HGTDBLPDataModule()
    elif dataset == Datasets.ACM:
        return HGTACMDataModule()
    else:
        return HGTIMDBDataModule()


def get_model(model: Models, datamodule: HGBBaseDataModule):
    if model == Models.HAN:
        return HAN(
            datamodule.metadata,
            in_channels=datamodule.in_channels,
            hidden_channels=256
        ), False
    elif model == Models.HGT:
        return HGT(
            datamodule.metadata,
            out_channels=datamodule.num_classes,
            hidden_channels=256
        ), False
    elif model == Models.HGCN:
        return HeteroGNN(
            datamodule.metadata,
            hidden_channels=256,
            out_channels=datamodule.num_classes,
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

        return to_hetero(gcn, datamodule.metadata), True
    else:
        gat = GAT(
            hidden_channels=256
        )
        return to_hetero_with_bases(gat, datamodule.metadata, num_bases=3,
                                    in_channels={'x': 64}), True


@hydra.main(version_base=None, config_path=".", config_name="nc_config")
def main(cfg: Config):
    if cfg.model == Models.HGT:
        datamodule = get_dataset_hgt(cfg.dataset)
    else:
        datamodule = get_dataset(cfg.dataset)

    datamodule.prepare_data()

    model, is_homegeneous = get_model(cfg.model, datamodule)

    classifier = NodeClassifier(model, hidden_channels=256, target=datamodule.target,
                                out_channels=datamodule.num_classes,
                                task=datamodule.task, homogeneous_model=is_homegeneous)

    # logger = WandbLogger(project="gnn-baselines", log_model=True)
    # logger.experiment.config["model"] = cfg.model
    # logger.experiment.config["dataset"] = cfg.dataset
    # logger.experiment.tags = ['GNN', 'baseline', 'node classification']
    # logger.log_hyperparams(
    #     {
    #         "n_heads": 8,
    #         "hidden_units": 256,
    #         "n_layers": 3,
    #         "optimiser": "AdamW"
    #     }
    # )

    trainer = L.Trainer(accelerator="cpu", log_every_n_steps=1,
                        fast_dev_run=True,
                        # logger=logger,
                        # strategy="ddp_find_unused_parameters_true",
                        # devices=4,
                        max_epochs=200,
                        callbacks=[EarlyStopping("valid/loss", patience=cfg.patience),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max", save_top_k=1)])
    trainer.fit(classifier, datamodule)
    trainer.test(classifier, datamodule)


if __name__ == '__main__':
    main()
