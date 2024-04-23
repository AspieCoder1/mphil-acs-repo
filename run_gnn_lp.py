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
from core.models import get_baseline_model, Models
from core.sheaf_configs import SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from models.recommender.recommender import GNNRecommender
from run_gnn_nc import ModelConfig


@dataclass
class Config:
    dataset: SheafLinkPredDatasetCfg
    model: ModelConfig
    trainer: TrainerArgs
    tags: list[str] = field(default_factory=list)
    hidden_dim: int = 64
    rec_metrics: bool = True


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="lp_config")
def main(cfg: Config):
    is_homogeneous = False
    if cfg.model.type == Models.GCN or cfg.model.type == Models.GAT:
        is_homogeneous = True

    datamodule = get_dataset_lp(cfg.dataset.name, is_homogeneous=is_homogeneous)
    datamodule.prepare_data()

    model, is_homogeneous = get_baseline_model(
        cfg.model.type, datamodule, hidden_channels=cfg.hidden_dim
    )

    link_predictor = GNNRecommender(
        model,
        edge_target=datamodule.target,
        homogeneous=is_homogeneous,
        batch_size=datamodule.batch_size,
        hidden_channels=cfg.hidden_dim,
        use_rec_metrics=cfg.rec_metrics,
        node_type_names=datamodule.node_type_names,
        edge_type_names=datamodule.edge_type_names,
    )

    logger = None
    checkpoint_name = "test_run"

    if cfg.trainer.logger:
        logger = WandbLogger(
            project="gnn-baselines", log_model=True, entity="acs-thesis-lb2027"
        )
        logger.experiment.config["model"] = cfg.model.type
        logger.experiment.config["dataset"] = cfg.dataset.name
        logger.experiment.tags = cfg.tags
        checkpoint_name = logger.version

    timer = Timer()

    trainer = L.Trainer(
        profiler=cfg.trainer.profiler,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        log_every_n_steps=1,
        max_epochs=cfg.trainer.max_epochs,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[
            EarlyStopping("valid/loss", patience=cfg.trainer.patience),
            ModelCheckpoint(
                dirpath=f"checkpoints/sheaflp_checkpoints/{checkpoint_name}",
                filename=cfg.model.type + "-" + cfg.dataset.name + "-{epoch}",
                monitor="valid/loss",
                mode="min",
                save_top_k=1,
            ),
            timer,
        ],
    )

    trainer.fit(link_predictor, datamodule)
    trainer.test(link_predictor, datamodule)

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
