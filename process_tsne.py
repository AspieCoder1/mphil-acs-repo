import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def main():
    print("loading singular values")
    singular_values = np.load("tsne-input/diag-dblp.npy")
    print("loading edge types")
    edge_types = np.load("tsne-input/diag-dblp-labels.npy")

    print("Loaded arrays")

    rng = np.random.default_rng(42)

    shuffled_idx = rng.permutation(np.arange(len(edge_types)))

    singular_values = singular_values[shuffled_idx][:20_000]
    edge_types = edge_types[shuffled_idx][:20_000]

    tsne_outputs = TSNE(n_components=2).fit_transform(
        singular_values, edge_types)

    # 5) Plotting the stuff
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=edge_types)
    fig.savefig("tsne-plots/tsne_diag_dblp.pdf", bbox_inches='tight', dpi=300)
    fig.savefig("tsne-plots/tsne_diag_dblp.png", bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
