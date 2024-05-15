#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass, field

import hydra
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from core.sheaf_configs import SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from link_prediction import LinkPredictor
from models.recommender.recommender import GNNRecommender
from run_gnn_nc import ModelConfig
from utils.instantiators import instantiate_loggers, instantiate_callbacks


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
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()

    model = hydra.utils.instantiate(
        cfg.model,
        in_channels=datamodule.in_channels,
        num_nodes=datamodule.num_nodes,
        num_relations=len(datamodule.metadata[1]),
        metadata=datamodule.metadata,
    )

    print(model)

    # model, is_homogeneous = get_baseline_model(
    #     cfg.model.type, datamodule, hidden_channels=cfg.hidden_dim
    # )

    link_predictor = LinkPredictor(
        model,
        edge_target=datamodule.target,
        homogeneous=cfg.dataset.homogeneous,
        batch_size=datamodule.batch_size,
        # hidden_channels=cfg.hidden_dim,
        # use_rec_metrics=False,
        node_type_names=datamodule.node_type_names,
        edge_type_names=datamodule.edge_type_names,
    )

    logger = instantiate_loggers(cfg.get("logger"))

    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{datamodule}"

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(link_predictor, datamodule)
    trainer.test(link_predictor, datamodule)

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
