import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from datasets.hgb import DBLPDataModule


def main() -> None:
    torch.set_float32_matmul_precision("high")

    # 1) get the datamodule
    datamodule = DBLPDataModule(homogeneous=True)
    datamodule.prepare_data()
    data: Data = datamodule.pyg_datamodule.data

    # 2) load checkpoint
    checkpoint = torch.load(
        "sheafnc_checkpoints/kj4z929k/DiagSheaf-DBLP-epoch=191.ckpt")
    print(checkpoint.keys())
    print(checkpoint['state_dict'])
    print(checkpoint['hparams'])
    # encoder = checkpoint["model"].to("cuda")
    #
    # # 3) calculate the singular values
    # x_maps = F.dropout(data.x, 0, training=False)
    # maps = encoder.sheaf_learners[0](x_maps.reshape(encoder.graph_size, -1),
    #                                  data.edge_index)
    # sdvals = torch.linalg.svdvals(maps).cpu().detach().numpy()
    # print(sdvals.shape)
    # tsne_outputs = TSNE(n_components=2).fit_transform(maps)
    #
    # # 4) Plotting the stuff
    # sns.set_style('whitegrid')
    # sns.set_context('paper')
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=data.edge_types)
    # fig.savefig("tsne_diag_dblp.pdf", bbox_inches='tight')
    # fig.savefig("tsne_diag_dblp.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
