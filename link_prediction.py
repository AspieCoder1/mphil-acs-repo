import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datasets.link_pred import LastFMDataModule
from models.HAN import HANLinkPredictor


def main():
    datamodule = LastFMDataModule("data")
    datamodule.prepare_data()

    print(datamodule.metadata)

    model = HANLinkPredictor(datamodule.metadata, hidden_channels=256,
                             edge_target=datamodule.target)

    logger = WandbLogger(project="gnn-baselines", log_model="all")
    logger.experiment.config["model"] = "HAN"
    logger.experiment.config["dataset"] = "LastFM"
    logger.experiment.tags = ['GNN', 'baseline', 'link_prediction']

    trainer = L.Trainer(accelerator="gpu", log_every_n_steps=1,
                        max_epochs=200,
                        fast_dev_run=True,
                        logger=logger,
                        callbacks=[EarlyStopping("valid/loss", patience=100),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max")])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
