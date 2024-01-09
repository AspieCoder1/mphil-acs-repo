from dataclasses import dataclass

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from core.datasets import get_dataset_lp, LinkPredDatasets
from core.models import Models, get_model
from core.trainer import TrainerArgs
from models import LinkPredictor


@dataclass
class Config:
    dataset: LinkPredDatasets
    model: Models
    trainer: TrainerArgs


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="lp_config")
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    datamodule = get_dataset_lp(cfg.dataset)
    datamodule.prepare_data()

    model, is_homogeneous = get_model(cfg.model, datamodule)

    link_predictor = LinkPredictor(model, edge_target=datamodule.target,
                                   homogeneous=is_homogeneous,
                                   batch_size=datamodule.batch_size)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model
    logger.experiment.config["dataset"] = cfg.dataset
    logger.experiment.tags = ['GNN', 'baseline', 'link_prediction']

    trainer = L.Trainer(log_every_n_steps=cfg.trainer.log_every_n_steps,
                        num_nodes=cfg.trainer.num_nodes,
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        strategy=cfg.trainer.strategy,
                        fast_dev_run=cfg.trainer.fast_dev_run,
                        max_epochs=200,
                        logger=logger,
                        callbacks=[
                            EarlyStopping("valid/loss", patience=cfg.trainer.patience),
                            ModelCheckpoint(monitor="valid/accuracy",
                                            mode="max", save_top_k=1)])

    trainer.fit(link_predictor, datamodule)
    trainer.test(link_predictor, datamodule)


if __name__ == '__main__':
    main()
