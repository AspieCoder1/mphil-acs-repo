#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Any

import lightning as L
from cuml import LogisticRegression
from cuml.metrics import accuracy
from cuml.model_selection import train_test_split

from models.sheaf_node_classifier import TrainStepOutput


class RestrictionMapCallback(L.Callback):
    def __init__(self):
        self.classifier = LogisticRegression()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
            outputs: TrainStepOutput,
        batch: Any,
        batch_idx: int,
    ) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            outputs['restriction_maps'], outputs['edge_types']
        )

        self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)

        acc = accuracy(y_test, preds)

        pl_module.log("train/restriction_map_accuracy", acc)
