from dataclasses import dataclass, field

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from core.datasets import get_dataset_nc
from core.sheaf_configs import SheafModelCfg, SheafNCDatasetCfg
from core.trainer import TrainerArgs
from models.SheafGNN.config import SheafModelArguments


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

    checkpoint = torch.load(
        "sheafnc_checkpoints/kj4z929k/DiagSheaf-DBLP-epoch=191.ckpt")
    encoder = checkpoint["encoder"].to("cuda")

    # 6) calculate the singular values
    x_maps = F.dropout(data.x, 0, training=False)
    maps = encoder.sheaf_learners[0](x_maps.reshape(encoder.graph_size, -1),
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


if __name__ == '__main__':
    main()
