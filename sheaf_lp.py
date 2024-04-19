#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import field, dataclass
from typing import Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import get_dataset_lp, LinkPredDatasets
from core.models import get_inductive_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from models.recommender.recommender import GNNRecommender
from models.sheaf_gnn.config import IndSheafModelArguments, SheafLearners
from sheaf_nc import init_sheaf_learner


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafLinkPredDatasetCfg = field(default_factory=SheafLinkPredDatasetCfg)
    model_args: IndSheafModelArguments = field(default_factory=IndSheafModelArguments)
    rec_metrics: bool = True
    sheaf_learner: SheafLearners = SheafLearners.local_concat


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config_lp")
def main(cfg: Config):
    dm = get_dataset_lp(LinkPredDatasets.LastFM, True)
    dm.prepare_data()

    cfg.model_args.graph_size = dm.graph_size
    cfg.model_args.input_dim = dm.in_channels
    cfg.model_args.output_dim = 64
    cfg.model_args.num_edge_types = dm.num_edge_types
    cfg.model_args.num_node_types = dm.num_node_types

    model_cls = get_inductive_sheaf_model(cfg.model.type)
    sheaf_learner = init_sheaf_learner(cfg)
    model = model_cls(None, cfg.model_args, sheaf_learner=sheaf_learner)

    print(dm.node_type_names)

    sheaf_lp = GNNRecommender(
        model=model,
        batch_size=dm.batch_size,
        hidden_channels=model.hidden_dim,
        edge_target=dm.target,
        homogeneous=True,
        use_rec_metrics=cfg.rec_metrics,
        node_type_names=dm.node_type_names,
        edge_type_names=dm.edge_type_names,
    )

    logger: Optional[WandbLogger] = None
    checkpoint_name = "test_run"
    if cfg.trainer.logger:
        logger = WandbLogger(
            project="gnn-baselines", log_model=True, entity="acs-thesis-lb2027"
        )
        logger.experiment.config["model"] = f"{cfg.model.type}-{cfg.sheaf_learner}"
        logger.experiment.config["dataset"] = cfg.dataset.name
        logger.experiment.tags = cfg.tags
        checkpoint_name = logger.version

    timer = Timer()

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=logger,
        precision="bf16",
        max_epochs=500,
        log_every_n_steps=1,
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

    trainer.fit(sheaf_lp, dm)
    trainer.test(sheaf_lp, dm)

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    if cfg.trainer.logger:
        logger.log_metrics(runtime)


if __name__ == "__main__":
    main()
