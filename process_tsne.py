import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def main():
    print("loading singular values")
    singular_values = np.load("tsne-input/diag-dblp.npy")
    print("loading edge types")
    edge_types = np.load("tsne-input/diag-dblp-labels.npy")

    print("Loaded arrays")

    tsne_outputs = PCA(n_components=2).fit_transform(
        singular_values)

    # 5) Plotting the stuff
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=edge_types)
    ax.legend(name="Edge type")
    fig.savefig("tsne-plots/pca_diag_dblp.pdf", bbox_inches='tight')
    fig.savefig("tsne-plots/pca_diag_dblp.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
