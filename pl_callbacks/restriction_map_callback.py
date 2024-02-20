#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Any

import lightning as L
from cuml import LogisticRegression
from cuml.metrics import accuracy
from cuml.model_selection import train_test_split
from lightning.pytorch.utilities.types import STEP_OUTPUT


class RestrictionMapCallback(L.Callback):
    def __init__(self):
        self.classifier = LogisticRegression()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        edge_types = trainer.train_dataloader().data.edge_types
        X_train, X_test, y_train, y_test = train_test_split(
            outputs["restriction_maps"], edge_types
        )

        self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)

        acc = accuracy(y_test, preds)

        pl_module.log("train/restriction_map_accuracy", acc)
