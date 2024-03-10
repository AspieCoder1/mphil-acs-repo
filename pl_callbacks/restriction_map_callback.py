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
from models.sheaf_node_classifier import TrainStepOutput


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
    def __init__(self, log_every_n_epoch: int, dataset: NCDatasets, model: ModelTypes):
        self.log_every_n_epoch: int = log_every_n_epoch
        self.dataset = dataset
        self.model = model

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

        unique_edge_types, unique_index = np.unique(edge_types, return_index=True)

        edge_types_to_label = {}

        edge = batch.edge_index[unique_index]
        src_type = batch.node_type[edge[:, 0]]
        dst_type = batch.node_type[edge[:, 1]]

        for i, edge_type in enumerate(unique_edge_types):
            src = src_type[i].item()
            dst = dst_type[i].item()
            edge_types_to_label[edge_type] = rf"{src: d} \to {dst: d}"

        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=edge_types,
            cmap="Spectral",
            s=3,
            rasterized=True,
        )
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.set_title(f"Epoch {pl_module.global_step}")
        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Edge types",
        )

        ax.add_artist(legend1)

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
