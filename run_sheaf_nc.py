#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Tuple, Optional

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import get_dataset_nc
from core.models import get_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafNCDatasetCfg
from core.trainer import TrainerArgs
from node_classification import SheafNodeClassifier
from models.sheaf_gnn.config import SheafModelArguments, SheafLearners
from models.sheaf_gnn.sheaf_models import (
    LocalConcatSheafLearner,
    TypeConcatSheafLearner,
    TypeEnsembleSheafLearner,
    NodeTypeConcatSheafLearner,
    EdgeTypeConcatSheafLearner,
    NodeTypeSheafLearner,
    EdgeTypeSheafLearner,
)
from pl_callbacks.restriction_map_callback import RestrictionMapUMAP


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafNCDatasetCfg = field(default_factory=SheafNCDatasetCfg)
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)
    sheaf_learner: SheafLearners = SheafLearners.type_ensemble
    plot_maps: bool = False


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = get_dataset_nc(cfg.dataset.name, True)
    datamodule.prepare_data()

    # 2) Update the config
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.num_edge_types = datamodule.num_edge_types
    cfg.model_args.num_node_types = datamodule.num_node_types
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    # 3) Initialise models
    model_cls = get_sheaf_model(cfg.model.type)
    sheaf_learner = init_sheaf_learner(cfg)

    model = model_cls(edge_index, cfg.model_args, sheaf_learner=sheaf_learner)

    sheaf_nc = SheafNodeClassifier(
        model,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task,
    )

    # 4) init trainer
    trainer, timer, logger = init_trainer(
        cfg, edge_type_names=datamodule.edge_type_names
    )

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) test the model
    trainer.test(sheaf_nc, datamodule)

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    if cfg.trainer.logger:
        logger.log_metrics(runtime)


def init_sheaf_learner(cfg):
    if cfg.sheaf_learner == SheafLearners.type_concat:
        sheaf_learner = TypeConcatSheafLearner
    elif cfg.sheaf_learner == SheafLearners.local_concat:
        sheaf_learner = LocalConcatSheafLearner
    elif cfg.sheaf_learner == SheafLearners.type_ensemble:
        sheaf_learner = TypeEnsembleSheafLearner
    elif cfg.sheaf_learner == SheafLearners.node_type_concat:
        sheaf_learner = NodeTypeConcatSheafLearner
    elif cfg.sheaf_learner == SheafLearners.node_type:
        sheaf_learner = NodeTypeSheafLearner
    elif cfg.sheaf_learner == SheafLearners.edge_type:
        sheaf_learner = EdgeTypeSheafLearner
    else:
        sheaf_learner = EdgeTypeConcatSheafLearner
    return sheaf_learner


def init_trainer(
    cfg: Config, edge_type_names: Optional[list[str]] = None
) -> Tuple[L.Trainer, Timer, WandbLogger]:
    logger = None
    checkpoint_name = "test_run"

    if cfg.trainer.logger:
        logger = WandbLogger(
            project="gnn-baselines",
            log_model=True,
            checkpoint_name=f"{cfg.model.type}-{cfg.dataset.name}",
            entity="acs-thesis-lb2027",
        )
        logger.experiment.config["model"] = f"{cfg.model.type}-{cfg.sheaf_learner}"
        logger.experiment.config["dataset"] = cfg.dataset.name
        logger.experiment.tags = cfg.tags
        checkpoint_name = logger.version
    timer = Timer(timedelta(hours=3))

    callbacks = [
        EarlyStopping("valid/loss", patience=cfg.trainer.patience),
        ModelCheckpoint(
            dirpath=f"checkpoints/sheafnc_checkpoints/{checkpoint_name}",
            filename=f"{cfg.model.type}-{cfg.dataset.name}",
            monitor="valid/accuracy",
            mode="max",
            save_top_k=1,
        ),
        timer,
    ]

    if cfg.plot_maps:
        callbacks.append(
            RestrictionMapUMAP(
                log_every_n_epoch=50,
                model=cfg.model.type,
                dataset=cfg.dataset.name,
                edge_type_names=edge_type_names,
            )
        )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    return trainer, timer, logger


if __name__ == "__main__":
    main()
