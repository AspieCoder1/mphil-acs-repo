import logging
from dataclasses import dataclass, field
from enum import auto
from typing import Type, Any

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from strenum import PascalCaseStrEnum

from core.datasets import NCDatasets, get_dataset_nc
from core.trainer import TrainerArgs
from models.SheafGNN import (
    DiscreteBundleSheafDiffusion,
    DiscreteDiagSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
    DiagSheafDiffusion,
    BundleSheafDiffusion,
    GeneralSheafDiffusion
)
from models.SheafGNN.config import SheafModelArguments
from models.SheafGNN.sheaf_base import SheafDiffusion
from models.SheafNodeClassifier import SheafNodeClassifier


class LogJobReturnCallback(Callback):
    """Log the job's return value or error upon job end"""

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
            self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")


class Model(PascalCaseStrEnum):
    DiagSheaf = auto()
    BundleSheaf = auto()
    GeneralSheaf = auto()
    DiagSheafODE = auto()
    BundleSheafODE = auto()
    GeneralSheafODE = auto()


def get_model(model: Model) -> Type[SheafDiffusion]:
    if model == Model.DiagSheaf:
        return DiscreteDiagSheafDiffusion
    if model == Model.BundleSheaf:
        return DiscreteBundleSheafDiffusion
    if model == Model.GeneralSheaf:
        return DiscreteGeneralSheafDiffusion
    if model == Model.DiagSheafODE:
        return DiagSheafDiffusion
    if model == Model.BundleSheafODE:
        return BundleSheafDiffusion
    if model == Model.GeneralSheafODE:
        return GeneralSheafDiffusion


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: Model = Model.BundleSheaf
    dataset: NCDatasets = NCDatasets.DBLP
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = get_dataset_nc(cfg.dataset, True)
    datamodule.prepare_data()

    # 2) Update the config
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    # 3) Initialise models
    model_cls = get_model(cfg.model)
    model = model_cls(edge_index, cfg.model_args)
    sheaf_nc = SheafNodeClassifier(
        model=model,
        hidden_channels=model.hidden_dim,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task
    )

    # 3.5) initialise logger
    logger = WandbLogger(log_model=True)
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
    logger.experiment.tags = cfg.tags

    # 4) initialise trainer
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=logger,
        max_epochs=200,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping("valid/loss",
                          patience=cfg.trainer.patience),
            ModelCheckpoint(dirpath="~/rds/hpc-work", monitor="valid/accuracy",
                            mode="max", save_top_k=1)]
    )

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) test the model
    trainer.test(sheaf_nc, datamodule)


if __name__ == '__main__':
    main()
