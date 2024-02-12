import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from umap import UMAP


def main():
    print("loading singular values")
    singular_values = np.load("tsne-input/diag-dblp.npy")
    print("loading edge types")
    edge_types = np.load("tsne-input/diag-dblp-labels.npy")

    print("Loaded arrays")

    rng = np.random.default_rng(42)

    shuffled_idx = rng.permutation(np.arange(len(edge_types)))

    singular_values = singular_values[shuffled_idx][:5_000]
    edge_types = edge_types[shuffled_idx][:5_000]

    umap_reducer = UMAP(random_state=42)
    embedding = umap_reducer.fit_transform(singular_values, edge_types)
    print("UMAP finished")

    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.scatter(embedding[:, 0], embedding[:, 1], c=edge_types, cmap='Spectral', s=5)
    fig.savefig("tsne-plots/umap_diag_dblp.pdf", bbox_inches='tight', dpi=300)
    fig.savefig("tsne-plots/umap_diag_dblp.png", bbox_inches='tight', dpi=300)
    print("Plotting finished")


if __name__ == '__main__':
    main()
