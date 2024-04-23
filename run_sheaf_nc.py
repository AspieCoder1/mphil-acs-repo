#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import List

import hydra
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger, Logger
from omegaconf import DictConfig

from node_classification import SheafNodeClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: DictConfig) -> None:
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    edge_index = datamodule.edge_index.to(cfg.model.args.device)

    model = hydra.utils.instantiate(
        cfg.model,
        edge_index=edge_index,
        args={
            "graph_size": datamodule.graph_size,
            "input_dim": datamodule.in_channels,
            "output_dim": datamodule.num_classes,
            "num_edge_types": datamodule.num_edge_types,
            "num_node_types": datamodule.num_node_types,
        },
    )

    sheaf_nc = SheafNodeClassifier(
        model,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task,
        homogeneous_model=True,
    )

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{datamodule}"

    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) test the model
    trainer.test(sheaf_nc, datamodule)

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
