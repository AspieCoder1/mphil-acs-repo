from dataclasses import dataclass
from enum import StrEnum

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datasets.hgb import DBLPDataModule, ACMDataModule, IMDBDataModule, \
    HGBBaseDataModule
from datasets.hgt import HGTDBLPDataModule, HGTACMDataModule, HGTIMDBDataModule, \
    HGTBaseDataModule
from models.HAN import HANEntityPredictor
from models.HGT import HGTEntityPredictor
from models.HeteroGNN import HeteroGNNNodeClassifier


class Datasets(StrEnum):
    DBLP = "DBLP"
    ACM = "ACM"
    IMDB = "IMDB"


class Models(StrEnum):
    HAN = "HAN"
    HGT = "HGT"
    HGCN = "HGCN"


@dataclass
class Config:
    dataset: Datasets
    model: Models
    patience: int = 100


cs = ConfigStore.instance()
cs.store("config", Config)


def get_dataset(dataset: Datasets) -> HGBBaseDataModule:
    match dataset:
        case Datasets.DBLP:
            return DBLPDataModule()
        case Datasets.ACM:
            return ACMDataModule()
        case Datasets.IMDB:
            return IMDBDataModule()


def get_dataset_hgt(dataset: Datasets) -> HGTBaseDataModule:
    match dataset:
        case Datasets.DBLP:
            return HGTDBLPDataModule()
        case Datasets.ACM:
            return HGTACMDataModule()
        case Datasets.IMDB:
            return HGTIMDBDataModule()


def get_model(model: Models, datamodule: HGBBaseDataModule):
    match model:
        case Models.HAN:
            return HANEntityPredictor(
                datamodule.metadata,
                out_channels=datamodule.num_classes,
                target=datamodule.target,
                task=datamodule.task
            )
        case Models.HGT:
            return HGTEntityPredictor(
                datamodule.metadata,
                out_channels=datamodule.num_classes,
                target=datamodule.target,
                task=datamodule.task
            )
        case Models.HGCN:
            return HeteroGNNNodeClassifier(
                datamodule.metadata,
                hidden_channels=256,
                out_channels=datamodule.num_classes,
                target=datamodule.target,
                task=datamodule.task,
                num_layers=3
            )


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    if cfg.model == Models.HGT:
        datamodule = get_dataset_hgt(cfg.dataset)
    else:
        datamodule = get_dataset(cfg.dataset)

    datamodule.prepare_data()

    model = get_model(cfg.model, datamodule)

    logger = WandbLogger(project="gnn-baselines", log_model="all")
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
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
                                                   mode="max")])
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
