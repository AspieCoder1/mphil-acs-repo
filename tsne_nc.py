from dataclasses import dataclass, field

import hydra
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from core.datasets import get_dataset_nc
from core.models import get_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafNCDatasetCfg
from core.trainer import TrainerArgs
from models.SheafGNN.config import SheafModelArguments
from models.SheafNodeClassifier import SheafNodeClassifier


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafNCDatasetCfg = field(default_factory=SheafNCDatasetCfg)
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")

    # 1) get the datamodule
    datamodule = get_dataset_nc(cfg.dataset.name, True)
    datamodule.prepare_data()
    data: Data = datamodule.pyg_datamodule.data

    # 2) Update the config
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    # 3) Initialise models
    model_cls = get_sheaf_model(cfg.model.type)
    model = model_cls(edge_index, cfg.model_args)

    sheaf_nc = SheafNodeClassifier(
        model,
        out_channels=datamodule.num_classes,
        target=datamodule.target,
        task=datamodule.task
    )

    # 4) init trainer
    trainer = init_trainer(cfg)

    # 5) train the model
    trainer.fit(sheaf_nc, datamodule)

    # 6) calculate the singular values
    x_maps = F.dropout(data.x, 0, training=False)
    maps = sheaf_nc.encoder.sheaf_learners[0](x_maps.reshape(model.graph_size, -1),
                                              data.edge_index)
    sdvals = torch.linalg.svdvals(maps).cpu().detach().numpy()
    print(sdvals.shape)
    tsne_outputs = TSNE(n_components=2).fit_transform(maps)

    # 7) Plotting the stuff
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=data.edge_types)
    fig.savefig("tsne_diag_dblp.pdf", bbox_inches='tight')
    fig.savefig("tsne_diag_dblp.png", bbox_inches='tight')


def init_trainer(cfg) -> L.Trainer:
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        fast_dev_run=cfg.trainer.fast_dev_run,
        precision="bf16-mixed",
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping("valid/loss",
                          patience=cfg.trainer.patience),
        ]
    )
    return trainer


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
