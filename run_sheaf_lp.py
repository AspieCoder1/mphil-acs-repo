#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from dataclasses import field, dataclass
from typing import List

import hydra
import torch
from hydra.core.config_store import ConfigStore
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger, Logger
from omegaconf import DictConfig

from core.sheaf_configs import SheafModelCfg, SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from link_prediction import SheafLinkPredictor
from models.sheaf_gnn.config import IndSheafModelArguments, SheafLearners
from utils.instantiators import instantiate_callbacks, instantiate_loggers


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
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    dm = hydra.utils.instantiate(cfg.dataset)
    dm.prepare_data()
    edge_index = dm.edge_index.to(cfg.model.args.device)

    model = hydra.utils.instantiate(
        cfg.model,
        edge_index=edge_index,
        args={
            "graph_size": dm.graph_size,
            "input_dim": cfg.get("input_dim", 64),
            "output_dim": 1,
            "num_edge_types": dm.num_edge_types,
            "num_node_types": dm.num_node_types,
        },
    )

    scheduler = hydra.utils.instantiate(cfg.scheduler, _partial_=True)
    optimiser = hydra.utils.instantiate(cfg.optimiser, _partial_=True)
    optimiser_name = cfg.optimiser['_target_'].split('.')[-1]

    if not scheduler:
        scheduler = None
        scheduler_name = 'ConstantLR'
    else:
        scheduler_name = cfg.scheduler['_target_'].split('.')[-1]

    decoder = hydra.utils.instantiate(cfg.decoder, dim=model.hidden_dim)

    sheaf_lp = SheafLinkPredictor(
        model=model,
        batch_size=dm.batch_size,
        hidden_dim=model.hidden_dim,
        in_feat=cfg.get("input_dim", 64),
        in_channels=dm.in_channels,
        target=dm.target,
        num_classes=1,
        scheduler=scheduler,
        optimiser=optimiser,
        decoder=decoder
    )


    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{dm}"
        logger[0].experiment.config['scheduler'] = scheduler_name
        logger[0].experiment.config['optimiser'] = optimiser_name

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 5) train the model
    trainer.fit(sheaf_lp, dm)

    # 6) test the model
    trainer.test(sheaf_lp, dm, ckpt_path='best')

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
