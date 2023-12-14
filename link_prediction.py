from dataclasses import dataclass
from enum import auto

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from strenum import PascalCaseStrEnum, UppercaseStrEnum

from datasets.link_pred import (
    LastFMDataModule,
    AmazonBooksDataModule,
    LinkPredBase
)
from models.HAN import HANLinkPredictor


class Datasets(PascalCaseStrEnum):
    LastFM = auto()
    AmazonBooks = auto()


class Models(UppercaseStrEnum):
    HAN = auto()


def get_dataset(dataset: Datasets) -> LinkPredBase:
    if dataset == Datasets.LastFM:
        return LastFMDataModule("data")
    else:
        return AmazonBooksDataModule("data")


@dataclass
class Config:
    dataset: Datasets
    model: Models
    patience: int = 100


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path=".", config_name="lp_config")
def main(cfg: Config):
    print(cfg)
    datamodule = get_dataset(cfg.dataset)
    print(datamodule)
    datamodule.prepare_data()
    print(datamodule.data)

    model = HANLinkPredictor(datamodule.metadata, hidden_channels=256,
                             edge_target=datamodule.target,
                             in_channels=datamodule.in_channels)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
    logger.experiment.tags = ['GNN', 'baseline', 'link_prediction']

    trainer = L.Trainer(log_every_n_steps=1,
                        num_nodes=1,
                        accelerator="gpu",
                        devices=1,
                        max_epochs=200,
                        logger=logger,
                        callbacks=[EarlyStopping("valid/loss", patience=100),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max", save_top_k=1)])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
