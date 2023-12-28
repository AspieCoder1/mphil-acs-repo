from dataclasses import dataclass

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from core.datasets import NCDatasets, get_dataset_nc, get_dataset_hgt
from core.models import Models, get_model
from core.trainer import TrainerArgs
from models import NodeClassifier


@dataclass
class Config:
    dataset: NCDatasets
    model: Models
    trainer: TrainerArgs


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path=".", config_name="nc_config")
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")

    if cfg.model == Models.HGT:
        datamodule = get_dataset_hgt(cfg.dataset)
    else:
        datamodule = get_dataset_nc(cfg.dataset)

    datamodule.prepare_data()

    model, is_homogeneous = get_model(cfg.model, datamodule.metadata)

    classifier = NodeClassifier(model, hidden_channels=256,
                                target=datamodule.target,
                                out_channels=datamodule.num_classes,
                                task=datamodule.task,
                                homogeneous_model=is_homogeneous)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
    logger.experiment.tags = ['GNN', 'baseline', 'node classification']
    logger.log_hyperparams(
        {
            "n_heads": 8,
            "hidden_units": 256,
            "n_layers": 3,
            "optimiser": "AdamW"
        }
    )

    trainer = L.Trainer(accelerator=cfg.trainer.accelerator, log_every_n_steps=1,
                        logger=logger,
                        devices=cfg.trainer.devices,
                        max_epochs=200,
                        callbacks=[
                            EarlyStopping("valid/loss",
                                          patience=cfg.trainer.patience),
                            ModelCheckpoint(monitor="valid/accuracy",
                                            mode="max", save_top_k=1)])
    trainer.fit(classifier, datamodule)
    trainer.test(classifier, datamodule)


if __name__ == '__main__':
    main()
