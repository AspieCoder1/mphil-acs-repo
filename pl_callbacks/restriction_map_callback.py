#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import lightning as L
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from models.sheaf_node_classifier import TrainStepOutput


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
        X_train, X_test, y_train, y_test = train_test_split(
            outputs['restriction_maps'].cpu().detach().numpy(),
            batch.edge_type.cpu().detach().numpy(),
        )

        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)

        pl_module.log("train/restriction_map_accuracy", acc, batch_size=1)
