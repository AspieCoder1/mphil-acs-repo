#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import List

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
    # dm.prepare_data()

    # initialise model
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        args={
            "num_features": 128,
        },
    )

    # initialise loggers
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # initialise callbacks
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # initialise trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
