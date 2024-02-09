import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
# from cuml import TSNE
from torch_geometric.data import Data
import numpy as np

from core.sheaf_configs import ModelTypes
from datasets.hgb import DBLPDataModule
from models.SheafGNN import DiscreteDiagSheafDiffusion
from models.SheafGNN.sheaf_base import SheafDiffusion
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

    # 3) calculate the restriction maps
    encoder: SheafDiffusion = model.encoder
    x = data.x.to(cfg.model_args.device)
    x = F.dropout(x, p=encoder.input_dropout, training=encoder.training)
    x = encoder.lin1(x)
    if encoder.use_act:
        x = F.elu(x)
    x = F.dropout(x, p=encoder.dropout, training=encoder.training)
    if encoder.second_linear:
        x = encoder.lin12(x)
    x = x.view(encoder.graph_size * encoder.final_d, -1)
    x_maps = F.dropout(x, 0, training=False)
    maps = encoder.sheaf_learners[0](x_maps.reshape(encoder.graph_size, -1),
                                     edge_index)


    # 4) calculate the singular values (only if not diagonal)
    if cfg.model.type != ModelTypes.DiagSheaf:
        singular_values = torch.linalg.svdvals(maps).cpu().detach().numpy()
    else:
        diag_sort, _ = torch.sort(torch.square(maps), dim=1, descending=True)
        singular_values = diag_sort.cpu().detach().numpy()

    np.save("tsne-input/diag-dblp.npy", singular_values)

    # tsne_outputs = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(
    #     singular_values)
    #
    # edge_types = torch.cat([data.edge_index, data.edge_type],
    #                        dim=-1).cpu().detach().numpy()
    #
    # # 5) Plotting the stuff
    # sns.set_style('whitegrid')
    # sns.set_context('paper')
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=edge_types)
    # ax.legend(name="Edge type")
    # fig.savefig("tsne_diag_dblp.pdf", bbox_inches='tight')
    # fig.savefig("tsne_diag_dblp.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
