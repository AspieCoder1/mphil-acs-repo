from dataclasses import dataclass, field

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import get_dataset_nc
from core.models import get_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafNCDatasetCfg
from core.trainer import TrainerArgs
from models import SheafNodeClassifier
from models.sheaf_gnn.config import SheafModelArguments


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafNCDatasetCfg = field(default_factory=SheafNCDatasetCfg)
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = get_dataset_nc(cfg.dataset.name, True)
    datamodule.prepare_data()

    # 2) Update the config
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    # 3) Initialise models

    model_cls = get_sheaf_model(cfg.model.type)
    model = model_cls(edge_index, cfg.model_args)

    sheaf_nc = SheafNodeClassifier(
        model,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task
    )

    # 4) init trainer
    trainer = init_trainer(cfg)

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) test the model
    trainer.test(sheaf_nc, datamodule)


def init_trainer(cfg) -> L.Trainer:
    logger = WandbLogger(
        project="gnn-baselines",
        log_model=True,
        checkpoint_name=f"{cfg.model_args.type}-{cfg.dataset.name}",
    )
    logger.experiment.config["model"] = cfg.model.type
    logger.experiment.config["dataset"] = cfg.dataset.name
    logger.experiment.tags = cfg.tags

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=logger,
        precision="bf16-mixed",
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping("valid/loss",
                          patience=cfg.trainer.patience),
            ModelCheckpoint(dirpath=f"checkpoints/sheafnc_checkpoints/{logger.version}",
                            filename=f'{cfg.model.type}-{cfg.dataset.name}',
                            monitor="valid/accuracy",
                            mode="max", save_top_k=1),
            Timer()
        ]
    )
    return trainer


if __name__ == '__main__':
    main()
