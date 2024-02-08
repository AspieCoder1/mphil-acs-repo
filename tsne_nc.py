import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from datasets.hgb import DBLPDataModule
from models.SheafGNN import DiscreteDiagSheafDiffusion
from models.SheafNodeClassifier import SheafNodeClassifier
from sheaf_nc import Config

cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")

    # 1) get the datamodule
    datamodule = DBLPDataModule(homogeneous=True)
    datamodule.prepare_data()
    data: Data = datamodule.pyg_datamodule.data

    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    encoder = DiscreteDiagSheafDiffusion(edge_index, cfg.model_args)

    model = SheafNodeClassifier.load_from_checkpoint(
        "sheafnc_checkpoints/kj4z929k/DiagSheaf-DBLP-epoch=191.ckpt",
        model=encoder
    )

    # 3) calculate the singular values
    x_maps = F.dropout(data.x, 0, training=False)
    maps = model.encoder.sheaf_learners[0](x_maps.reshape(model.encoder.graph_size, -1),
                                           edge_index)
    sdvals = torch.linalg.svdvals(maps).cpu().detach().numpy()
    print(sdvals.shape)
    tsne_outputs = TSNE(n_components=2).fit_transform(maps)

    # 4) Plotting the stuff
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=data.edge_types)
    fig.savefig("tsne_diag_dblp.pdf", bbox_inches='tight')
    fig.savefig("tsne_diag_dblp.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
