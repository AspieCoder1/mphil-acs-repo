from dataclasses import dataclass, field
from enum import auto
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from strenum import PascalCaseStrEnum
import lightning as L

from core.datasets import NCDatasets, get_dataset_nc
from core.trainer import TrainerArgs
from models.SheafGNN import DiscreteBundleSheafDiffusion
from models.SheafGNN.config import SheafModelArguments
from models.SheafNodeClassifier import SheafNodeClassifier


class Model(PascalCaseStrEnum):
    DiagSheaf = auto()
    BundleSheaf = auto()
    GeneralSheaf = auto()
    DiagSheafODE = auto()
    BundleSheafODE = auto()
    GeneralSheafODE = auto()


@dataclass
class Config:
    model: Model
    dataset: NCDatasets
    trainer: TrainerArgs
    defaults: list[Any] = field(default_factory=lambda: ["_self_"])
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="sheaf_config.yaml")
def main(cfg: Config) -> None:
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = get_dataset_nc(cfg.dataset, True)
    datamodule.prepare_data()

    # 2) Update the config
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes


    # 3) Initialise models
    model = DiscreteBundleSheafDiffusion(datamodule.edge_index, cfg.model_args)
    sheaf_nc = SheafNodeClassifier(
        model=model,
        hidden_channels=model.hidden_dim,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task
    )

    # 4) initialise trainer
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        # fast_dev_run=True,
        max_epochs=200,
        callbacks=[
            EarlyStopping("valid/loss",
                          patience=cfg.trainer.patience),
            ModelCheckpoint(monitor="valid/accuracy",
                            mode="max", save_top_k=1)]
    )

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) test the model
    trainer.test(sheaf_nc, datamodule)


if __name__ == '__main__':
    main()
