#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import lightning as L
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from models.sheaf_node_classifier import TrainStepOutput


class RestrictionMapCallback(L.Callback):
    def __init__(self):
        self.classifier = LogisticRegression()

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
            batch.edge_types.cpu().detach().numpy()
        )

        self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)

        acc = accuracy_score(y_test, preds)

        pl_module.log("train/restriction_map_accuracy", acc, batch_size=1)
