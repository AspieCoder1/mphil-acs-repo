#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.loggers import WandbLogger, Logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from typing_extensions import TypeGuard, Protocol
from umap import UMAP

from core.datasets import NCDatasets
from core.sheaf_configs import ModelTypes
from node_classification.sheaf_node_classifier import TrainStepOutput


class ProcessesRestrictionMaps(Protocol):
    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor: ...


def is_sheaf_encoder(module: L.LightningModule) -> TypeGuard[ProcessesRestrictionMaps]:
    if not hasattr(module, "encoder"):
        return False

    if not hasattr(module.encoder, "process_restriction_maps"):
        return False
    return True


def is_wandb_logger(module: Logger) -> TypeGuard[WandbLogger]:
    return isinstance(module, WandbLogger)


class RestrictionMapCallback(L.Callback):
    def __init__(self):
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1_000),
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: TrainStepOutput,
        batch: Data,
        batch_idx: int,
    ) -> None:
        if not is_sheaf_encoder(pl_module):
            return None

        restriction_maps = pl_module.encoder.process_restriction_maps(
            outputs["restriction_maps"]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            restriction_maps.cpu().detach().numpy(),
            batch.edge_type.cpu().detach().numpy(),
        )

        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)

        pl_module.log("train/restriction_map_accuracy", acc, batch_size=1)


class RestrictionMapUMAP(L.Callback):
    def __init__(
        self,
        log_every_n_epoch: int,
        dataset: NCDatasets,
        model: ModelTypes,
        edge_type_names: list[str],
    ):
        self.log_every_n_epoch: int = log_every_n_epoch
        self.dataset = dataset
        self.model = model
        self.edge_type_names = edge_type_names

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: TrainStepOutput,
        batch: Data,
        batch_idx: int,
    ) -> None:

        if (
            pl_module.global_step % self.log_every_n_epoch != 0
            and pl_module.global_step != 1
        ):
            return None

        if not is_sheaf_encoder(pl_module):
            return None

        edge_types = batch.edge_type.cpu().detach().numpy()
        sample_idx, _, edge_types, _ = train_test_split(
            np.arange(len(edge_types)),
            edge_types,
            stratify=edge_types,
            random_state=42,
            train_size=0.2,
        )

        restriction_maps = outputs["restriction_maps"][sample_idx]

        restriction_maps = (
            pl_module.encoder.process_restriction_maps(restriction_maps)
            .cpu()
            .detach()
            .numpy()
        )

        umap = UMAP()
        embeddings = umap.fit_transform(restriction_maps)

        sns.set_style("whitegrid")
        sns.set_context("paper")
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        if not os.path.exists(f"umap-plots/{self.model}/{self.dataset}"):
            os.makedirs(f"umap-plots/{self.model}/{self.dataset}", exist_ok=True)

        for i, edge_type in enumerate(self.edge_type_names):
            edge_mask = edge_types == i
            embs = embeddings[edge_mask]

            ax.scatter(
                embs[:, 0],
                embs[:, 1],
                label=rf"{edge_type[0]}$\to${edge_type[-1]}",
                s=3,
                rasterized=True,
            )
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.set_title(f"Epoch {pl_module.global_step}")
        ax.legend()

        plt.savefig(
            f"umap-plots/{self.model}/{self.dataset}/step-{pl_module.global_step}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            f"umap-plots/{self.model}/{self.dataset}/step-{pl_module.global_step}.png",
            dpi=300,
            bbox_inches="tight",
        )

        logger = trainer.logger
        if is_wandb_logger(logger):
            logger.experiment.log({"UMAP Plot": fig})
