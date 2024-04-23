#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import hydra
from lightning import LightningDataModule, Callback, Trainer, LightningModule
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger

from utils.instantiators import instantiate_loggers, instantiate_callbacks


@hydra.main(version_base=None, config_path="configs", config_name="dti_config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    # initialise data module
    dm: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    dm.prepare_data()

    # initialise model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    print(model)

    # initialise loggers
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    print(logger)

    # initialise callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    print(callbacks)

    # initialise trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    print(trainer)


if __name__ == "__main__":
    main()
