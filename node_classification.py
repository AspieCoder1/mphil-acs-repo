#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import NCDatasets, get_dataset_nc, get_dataset_hgt
from core.models import Models, get_baseline_model
from core.trainer import TrainerArgs
from models import NodeClassifier


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
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")

    print(cfg.model.type)

    if cfg.model.type == Models.HGT:
        datamodule = get_dataset_hgt(cfg.dataset.name)
    elif cfg.model.type == Models.GCN or cfg.model.type == Models.GAT:
        print("Should print")
        datamodule = get_dataset_nc(cfg.dataset.name, homogeneous=True)
    else:
        print("Should not print")
        datamodule = get_dataset_nc(cfg.dataset.name)

    datamodule.prepare_data()
    print(datamodule.in_channels)

    model, is_homogeneous = get_baseline_model(cfg.model.type, datamodule)

    classifier = NodeClassifier(
        model,
        hidden_channels=256,
        target=datamodule.target,
        out_channels=datamodule.num_classes,
        task=datamodule.task,
        homogeneous_model=is_homogeneous,
    )

    logger = None
    timer = Timer()
    checkpoint_name = "test_run"

    if cfg.trainer.logger:
        logger = WandbLogger(
            project="gnn-baselines",
            log_model=True,
            save_dir="~/rds/hpc-work/.wandb",
            entity="acs-thesis-lb2027",
        )
        logger.experiment.config["model"] = cfg.model.type
        logger.experiment.config["dataset"] = cfg.dataset.name
        logger.experiment.tags = cfg.tags
        checkpoint_name = logger.version

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=1,
        logger=logger,
        devices=cfg.trainer.devices,
        fast_dev_run=cfg.trainer.fast_dev_run,
        max_epochs=200,
        callbacks=[
            EarlyStopping("valid/loss", patience=cfg.trainer.patience),
            ModelCheckpoint(
                dirpath=f"gnn_nc_checkpoints/{checkpoint_name}",
                monitor="valid/accuracy",
                mode="max",
                save_top_k=1,
            ),
            timer,
        ],
    )
    trainer.fit(classifier, datamodule)
    trainer.test(classifier, datamodule)

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    if cfg.trainer.logger:
        logger.log_metrics(runtime)
    else:
        print(runtime)


if __name__ == "__main__":
    main()
