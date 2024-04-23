#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import hydra
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from core.models import get_sheaf_model
from models.sheaf_gnn.config import SheafLearners
from models.sheaf_gnn.sheaf_models import (
    LocalConcatSheafLearner,
    TypeConcatSheafLearner,
    TypeEnsembleSheafLearner,
    NodeTypeConcatSheafLearner,
    EdgeTypeConcatSheafLearner,
    NodeTypeSheafLearner,
    EdgeTypeSheafLearner,
)
from node_classification import SheafNodeClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: DictConfig) -> None:
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()

    model_args = hydra.utils.instantiate(cfg.model_args)

    # 2) Update the config
    model_args.graph_size = datamodule.graph_size
    model_args.input_dim = datamodule.in_channels
    model_args.output_dim = datamodule.num_classes
    model_args.graph_size = datamodule.graph_size
    model_args.num_edge_types = datamodule.num_edge_types
    model_args.num_node_types = datamodule.num_node_types
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    # 3) Initialise models
    model_cls = get_sheaf_model(cfg.model.type)
    sheaf_learner = init_sheaf_learner(cfg)

    model = model_cls(edge_index, model_args, sheaf_learner=sheaf_learner)

    sheaf_nc = SheafNodeClassifier(
        model,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task,
        homogeneous_model=True,
    )

    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
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

    if len(logger) > 0:
        logger[0].log_metrics(runtime)
    else:
        print(runtime)


def init_sheaf_learner(cfg):
    if cfg["sheaf_learner"] == SheafLearners.type_concat:
        sheaf_learner = TypeConcatSheafLearner
    elif cfg["sheaf_learner"] == SheafLearners.local_concat:
        sheaf_learner = LocalConcatSheafLearner
    elif cfg["sheaf_learner"] == SheafLearners.type_ensemble:
        sheaf_learner = TypeEnsembleSheafLearner
    elif cfg["sheaf_learner"] == SheafLearners.node_type_concat:
        sheaf_learner = NodeTypeConcatSheafLearner
    elif cfg["sheaf_learner"] == SheafLearners.node_type:
        sheaf_learner = NodeTypeSheafLearner
    elif cfg["sheaf_learner"] == SheafLearners.edge_type:
        sheaf_learner = EdgeTypeSheafLearner
    else:
        sheaf_learner = EdgeTypeConcatSheafLearner
    return sheaf_learner


# def init_trainer(
#     cfg: Dic, edge_type_names: Optional[list[str]] = None
# ) -> Tuple[L.Trainer, Timer, WandbLogger]:
#     logger = None
#     checkpoint_name = "test_run"
#
#     if cfg.trainer.logger:
#         logger = WandbLogger(
#             project="gnn-baselines",
#             log_model=True,
#             checkpoint_name=f"{cfg.model.type}-{cfg.dataset.name}",
#             entity="acs-thesis-lb2027",
#         )
#         logger.experiment.config["model"] = f"{cfg.model.type}-{cfg['sheaf_learner']}"
#         logger.experiment.config["dataset"] = cfg.dataset.name
#         logger.experiment.tags = cfg.tags
#         checkpoint_name = logger.version
#     timer = Timer(timedelta(hours=3))
#
#     callbacks = [
#         EarlyStopping("valid/loss", patience=cfg.trainer.patience),
#         ModelCheckpoint(
#             dirpath=f"checkpoints/sheafnc_checkpoints/{checkpoint_name}",
#             filename=f"{cfg.model.type}-{cfg.dataset.name}",
#             monitor="valid/accuracy",
#             mode="max",
#             save_top_k=1,
#         ),
#         timer,
#     ]
#
#     if cfg.plot_maps:
#         callbacks.append(
#             RestrictionMapUMAP(
#                 log_every_n_epoch=50,
#                 model=cfg.model.type,
#                 dataset=cfg.dataset.name,
#                 edge_type_names=edge_type_names,
#             )
#         )
#
#     trainer = L.Trainer(
#         accelerator=cfg.trainer.accelerator,
#         devices=cfg.trainer.devices,
#         num_nodes=cfg.trainer.num_nodes,
#         strategy=cfg.trainer.strategy,
#         fast_dev_run=cfg.trainer.fast_dev_run,
#         logger=logger,
#         max_epochs=cfg.trainer.max_epochs,
#         log_every_n_steps=1,
#         callbacks=callbacks,
#     )
#     return trainer, timer, logger


if __name__ == "__main__":
    main()
