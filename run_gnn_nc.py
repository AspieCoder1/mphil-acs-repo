#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from typing import Union, List

import hydra
from hydra.core.config_store import ConfigStore
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

from core.datasets import NCDatasets
from core.models import Models
from core.trainer import TrainerArgs
from datasets.hgb import HGBBaseDataModule
from datasets.hgt import HGTBaseDataModule
from node_classification import NodeClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks


@dataclass
class ModelConfig:
    type: Models = Models.GCN


@dataclass
class DatasetConfig:
    name: NCDatasets = NCDatasets.DBLP


@dataclass
class Config:
    tags: list[str]
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerArgs


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="nc_config")
def main(cfg: DictConfig):
    datamodule: Union[HGTBaseDataModule, HGBBaseDataModule] = hydra.utils.instantiate(
        cfg.dataset
    )

    datamodule.prepare_data()

    model = hydra.utils.instantiate(
        cfg.model,
        in_channels=datamodule.in_channels,
        num_nodes=datamodule.num_nodes,
        num_relations=len(datamodule.metadata[1]),
        metadata=datamodule.metadata,
    )

    classifier = NodeClassifier(
        model,
        hidden_channels=256,
        target=datamodule.target,
        out_channels=datamodule.num_classes,
        task=datamodule.task,
        homogeneous_model=cfg.dataset.homogeneous,
    )

    print(f'{model}-{datamodule}')

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{datamodule}"

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 5) train the model
    trainer.fit(classifier, datamodule)

    # 6) test the model
    trainer.test(classifier, datamodule)

    timer = next(filter(lambda x: isinstance(x, Timer), callbacks))

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    if logger:
        trainer.logger.log_metrics(runtime)
    else:
        print(runtime)


if __name__ == "__main__":
    main()
