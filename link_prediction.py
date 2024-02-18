#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass, field

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import get_dataset_lp
from core.models import get_model
from core.sheaf_configs import SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from models import LinkPredictor
from node_classification import ModelConfig


@dataclass
class Config:
    dataset: SheafLinkPredDatasetCfg
    model: ModelConfig
    trainer: TrainerArgs
    tags: list[str] = field(default_factory=list)


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="lp_config")
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    datamodule = get_dataset_lp(cfg.dataset.name)
    datamodule.prepare_data()

    model, is_homogeneous = get_model(cfg.model.type, datamodule)

    link_predictor = LinkPredictor(model, edge_target=datamodule.target,
                                   homogeneous=is_homogeneous,
                                   batch_size=datamodule.batch_size)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model.type
    logger.experiment.config["dataset"] = cfg.dataset.name
    logger.experiment.tags = cfg.tags

    timer = Timer()

    trainer = L.Trainer(num_nodes=cfg.trainer.num_nodes,
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        strategy=cfg.trainer.strategy,
                        fast_dev_run=cfg.trainer.fast_dev_run,
                        log_every_n_steps=1,
                        max_epochs=200,
                        logger=logger,
                        callbacks=[
                            EarlyStopping("valid/loss", patience=cfg.trainer.patience),
                            ModelCheckpoint(monitor="valid/HR@20",
                                            mode="max", save_top_k=1),
                            timer
                        ])

    trainer.fit(link_predictor, datamodule)
    trainer.test(link_predictor, datamodule)

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }
    print(runtime)
    logger.log_metrics(runtime)


if __name__ == '__main__':
    main()
