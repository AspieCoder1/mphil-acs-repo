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
                             edge_target=datamodule.target,
                             in_channels=datamodule.in_channels)

    logger = WandbLogger(project="gnn-baselines", log_model=True)
    logger.experiment.config["model"] = "HAN"
    logger.experiment.config["dataset"] = "LastFM"
    logger.experiment.tags = ['GNN', 'baseline', 'link_prediction']

    trainer = L.Trainer(log_every_n_steps=1,
                        num_nodes=1,
                        accelerator="gpu",
                        devices=4,
                        strategy="fsdp",
                        max_epochs=200,
                        logger=logger,
                        callbacks=[EarlyStopping("valid/loss", patience=100),
                                   ModelCheckpoint(monitor="valid/accuracy",
                                                   mode="max", save_top_k=1)])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
