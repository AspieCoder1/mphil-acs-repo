from dataclasses import dataclass
from enum import auto

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from strenum import UppercaseStrEnum

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
    GATNodeClassifier,
    GCNNodeClassifier,
    HANNodeClassifier,
    HGTEntityPredictor,
    HeteroGNNNodeClassifier,
    RGCNNodeClassifier
)


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
        return HANNodeClassifier(
            datamodule.metadata,
            in_channels=datamodule.in_channels,
            out_channels=datamodule.num_classes,
            target=datamodule.target,
            task=datamodule.task
        )
    elif model == Models.HGT:
        return HGTEntityPredictor(
            datamodule.metadata,
            out_channels=datamodule.num_classes,
            target=datamodule.target,
            task=datamodule.task
        )
    elif model == Models.HGCN:
        return HeteroGNNNodeClassifier(
            datamodule.metadata,
            hidden_channels=256,
            out_channels=datamodule.num_classes,
            target=datamodule.target,
            task=datamodule.task,
            num_layers=3
        )
    elif model == Models.RGCN:
        return RGCNNodeClassifier(
            metadata=datamodule.metadata,
            hidden_channels=256,
            out_channels=datamodule.num_classes,
            num_nodes=datamodule.num_nodes,
            num_relations=len(datamodule.metadata[1]),
            task=datamodule.task,
            target=datamodule.target
        )
    elif model == Models.GCN:
        return GCNNodeClassifier(
            datamodule.metadata,
            out_channels=datamodule.num_classes,
            target=datamodule.target,
            task=datamodule.task
        )
    else:
        return GATNodeClassifier(
            datamodule.metadata,
            out_channels=datamodule.num_classes,
            target=datamodule.target,
            task=datamodule.task
        )


@hydra.main(version_base=None, config_path=".", config_name="nc_config")
def main(cfg: Config):
    if cfg.model == Models.HGT:
        datamodule = get_dataset_hgt(cfg.dataset)
    else:
        datamodule = get_dataset(cfg.dataset)

    datamodule.prepare_data()

    model = get_model(cfg.model, datamodule)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
    logger.experiment.tags = ['GNN', 'baseline', 'node classification']
    logger.log_hyperparams(
        {
            "n_heads": 8,
            "hidden_units": 256,
            "n_layers": 3,
            "optimiser": "AdamW"
        }
    )

    trainer = L.Trainer(accelerator="cpu", log_every_n_steps=1,
                        logger=logger,
                        max_epochs=200,
                        callbacks=[EarlyStopping("valid/loss", patience=cfg.patience),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max", save_top_k=1)])
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
