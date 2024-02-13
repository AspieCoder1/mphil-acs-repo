import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from umap import UMAP


def main():
    print("loading singular values")
    singular_values = np.load("tsne-input/GeneralSheaf-DBLP.npy")
    print("loading edge types")
    edge_types = np.load("tsne-input/GeneralSheaf-DBLP-labels.npy")

    print("Loaded arrays")

    rng = np.random.default_rng(42)

    shuffled_idx = rng.permutation(np.arange(len(edge_types)))

    singular_values = singular_values[shuffled_idx][:5_000]
    edge_types = edge_types[shuffled_idx][:5_000]

    umap_reducer = UMAP(random_state=42, min_dist=0.0, n_neighbors=1_000)
    embedding = umap_reducer.fit_transform(singular_values, edge_types)
    print("UMAP finished")

    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(embedding[:, 0], embedding[:, 1], c=edge_types, cmap='Spectral', s=5)
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    fig.savefig("tsne-plots/umap_general_dblp.pdf", bbox_inches='tight', dpi=300)
    fig.savefig("tsne-plots/umap_general_dblp.png", bbox_inches='tight', dpi=300)
    print("Plotting finished")


if __name__ == '__main__':
    main()
