from dataclasses import dataclass
from enum import auto

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from strenum import PascalCaseStrEnum

from datasets.link_pred import (
    LastFMDataModule,
    AmazonBooksDataModule,
    LinkPredBase
)
from models import LinkPredictor
from node_classification import get_model, Models


class Datasets(PascalCaseStrEnum):
    LastFM = "LastFM"
    AmazonBooks = auto()


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

    model, is_homogeneous = get_model(cfg.model, datamodule)

    link_predictor = LinkPredictor(model, edge_target=datamodule.target,
                                   homogeneous=is_homogeneous)

    # logger = WandbLogger(project="gnn-baselines", log_model=True)
    # logger.experiment.config["model"] = cfg.model
    # logger.experiment.config["dataset"] = cfg.dataset
    # logger.experiment.tags = ['GNN', 'baseline', 'link_prediction']

    trainer = L.Trainer(log_every_n_steps=1,
                        num_nodes=1,
                        accelerator="cpu",
                        devices=1,
                        max_epochs=200,
                        fast_dev_run=True,
                        # logger=logger,
                        callbacks=[EarlyStopping("valid/loss", patience=100),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max", save_top_k=1)])

    trainer.fit(link_predictor, datamodule)
    # trainer.test(link_predictor, datamodule)


if __name__ == '__main__':
    main()
