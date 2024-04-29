#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import List

import hydra
from lightning import LightningDataModule, Callback, LightningModule, Trainer
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger

from dti_prediction.sheaf_models import DTIPredictionModule
from utils.instantiators import instantiate_loggers, instantiate_callbacks


@hydra.main(version_base=None, config_path="configs", config_name="dti_config")
def main(cfg: DictConfig) -> None:
    # initialise data module
    dm: LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    # initialise model
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
    )

    dti_predictor = DTIPredictionModule(
        model=model, use_score_function=cfg.use_score_func, out_channels=64
    )

    # initialise loggers
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # initialise callbacks
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # initialise trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    trainer.fit(dti_predictor, dm)
    trainer.test(dti_predictor, dm)


if __name__ == "__main__":
    main()
