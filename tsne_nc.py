import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from datasets.hgb import DBLPDataModule
from models.SheafNodeClassifier import SheafNodeClassifier


def main():
    dm = DBLPDataModule(homogeneous=True)
    dm.prepare_data()
    data: Data = dm.pyg_datamodule.data
    model = SheafNodeClassifier.load_from_checkpoint(
        "sheafnc_checkpoints/kj4z929k/DiagSheaf-DBLP-epoch=191.ckpt")
    x_maps = F.dropout(data.x, 0, training=False)
    maps = model.encoder.sheaf_learners[0](x_maps.reshape(model.graph_size, -1),
                                           data.edge_index)
    sdvals = torch.linalg.svdvals(maps).cpu().detach().numpy()
    print(sdvals.shape)
    tsne_outputs = TSNE(n_components=2).fit_transform(maps)

    # Plotting the stuff
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=data.edge_types)
    fig.savefig("tsne_diag_dblp.pdf", bbox_inches='tight')
    fig.savefig("tsne_diag_dblp.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
