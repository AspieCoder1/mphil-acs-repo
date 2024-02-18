#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.config_store import ConfigStore
from umap import UMAP

from core.datasets import NCDatasets
from core.sheaf_configs import SheafModelCfg


@dataclass
class Config:
    model: SheafModelCfg
    dataset: NCDatasets


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="process_umap.yaml")
def main(cfg: Config):
    print("loading singular values")
    singular_values = np.load(f"umap-input/{cfg.model.type}-{cfg.dataset.name}.npy")
    print("loading edge types")
    edge_types = np.load(f"umap-input/{cfg.model.type}-{cfg.dataset.name}-labels.npy")

    print("Loaded arrays")

    rng = np.random.default_rng(42)

    shuffled_idx = rng.permutation(np.arange(len(edge_types)))

    singular_values = singular_values[shuffled_idx][:5_000]
    edge_types = edge_types[shuffled_idx][:5_000]

    umap_reducer = UMAP(random_state=42, min_dist=0.0, n_neighbors=2_500)
    embedding = umap_reducer.fit_transform(singular_values, edge_types)
    print("UMAP finished")

    sns.set_style('whitegrid')
    sns.set_context('paper')
    cm = 1 / 2.54
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    ax.scatter(embedding[:, 0], embedding[:, 1], c=edge_types, cmap='Spectral', s=3)
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    fig.savefig(f"umap-plots/{cfg.model.type}-{cfg.dataset.name}.pdf",
                bbox_inches='tight',
                dpi=300)
    fig.savefig(f"umap-plots/{cfg.model.type}-{cfg.dataset.name}.png",
                bbox_inches='tight',
                dpi=300)
    print("Plotting finished")


if __name__ == '__main__':
    main()
