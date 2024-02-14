from dataclasses import field, dataclass

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from core.datasets import get_dataset_lp, LinkPredDatasets
from core.models import get_inductive_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from models.SheafGNNInductive.config import SheafModelArguments
from models.SheafLinkPredictor import SheafLinkPredictor


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafLinkPredDatasetCfg = field(default_factory=SheafLinkPredDatasetCfg)
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config_lp")
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    dm = get_dataset_lp(LinkPredDatasets.LastFM, True)
    dm.prepare_data()

    cfg.model_args.graph_size = dm.graph_size
    cfg.model_args.input_dim = dm.in_channels
    cfg.model_args.output_dim = 64

    model_cls = get_inductive_sheaf_model(cfg.model.type)
    model = model_cls(None, cfg.model_args)

    sheaf_lp = SheafLinkPredictor(
        model=model, num_classes=1,
        hidden_dim=model.hidden_dim
    )

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = cfg.model.type
    logger.experiment.config["dataset"] = cfg.dataset.name
    logger.experiment.tags = cfg.tags

    timer = Timer()

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
            ModelCheckpoint(dirpath=f"sheaflp_checkpoints/{logger.version}",
                            filename=cfg.model.type + '-' + cfg.dataset.name + '-{epoch}',
                            monitor="valid/accuracy",
                            mode="max", save_top_k=1),
            timer
        ]
    )

    trainer.fit(sheaf_lp, dm)
    trainer.test(sheaf_lp, dm)

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    logger.log_metrics(runtime)


if __name__ == "__main__":
    main()
