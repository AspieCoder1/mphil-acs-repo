#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os.path as osp
from pathlib import Path
from typing import Union, List, Tuple, Literal

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.nn.models import Node2Vec
from torch_geometric.typing import Adj, FeatureTensorType

import utils

EDGE_TYPE_MAP = {
    ("drug", "disease"): "drug_treats",
    ("drug", "protein"): "drug_targets",
    ("protein", "drug"): "protein_interacts_with",
    ("protein", "disease"): "protein_linked_to",
    ("disease", "drug"): "disease_treated_by",
    ("disease", "protein"): "disease_linked_to",
}
EDGE_TYPE_NAMES = [
    "drug_treats",
    "drug_targets",
    "protein_interacts_with",
    "protein_linked_to",
    "disease_treated_by",
    "disease_linked_to",
]
NODE_TYPE_NAMES = ["drug", "protein", "disease"]


class DTIDataset(InMemoryDataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset: Literal["deepDTnet_20", "KEGG_MED", "DTINet_17"] = "deepDTNet_20",
    ):
        # super(DTIDataset, self).__init__(root_dir, transform, pre_transform)
        self.dataset = dataset
        self.edge_type_map = EDGE_TYPE_MAP
        self.edge_type_names = EDGE_TYPE_NAMES
        self.node_type_names = NODE_TYPE_NAMES
        self.nodes_per_type: dict[str, int] = {}
        self.hyperedges_per_type: dict[str, int] = {}
        self.file_ids: dict[str, str] = {
            "deepDTnet_20": "1RGS2K58Gjr5IxPJTE4G-MHl0S6Wk6UgZ",
            "KEGG_MED": "1_XOT7Czd560UvkxpJM1-L5t9GXDPLhQr",
            "DTINet_17": "1pLoNyznbcTaxBHW8cSNPUU6oN3WCAh3l",
        }
        self.transform = transform
        self.pre_transform = pre_transform
        super().__init__(root_dir, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["drug_disease.txt", "drug_protein.txt", "protein_disease.txt"]

    @property
    def num_nodes(self):
        return sum(self.nodes_per_type.values())

    @property
    def num_hyperedges(self):
        return sum(self.hyperedges_per_type.values())

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.dataset, "raw")

    @property
    def raw_paths(self) -> List[str]:
        return [osp.join(self.raw_dir, filename) for filename in self.raw_file_names]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return "data.pt"

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.dataset, "processed")

    def download(self):
        url = f"https://drive.google.com/uc?export=download&id={self.file_ids[self.dataset]}"
        path = download_url(url=url, folder=self.raw_dir, filename="data.zip")
        extract_zip(path, self.raw_dir)

    def generate_incidence_graph(self, hyperedge_index: Adj) -> Adj:
        """
        Generates the incidence graph of a hypergraph.

        Given a `hyperedge_index` we convert the hypergraph into a bipartite incidence
        graph representation.

        Args:
            hyperedge_index (Adj): edge index of the input hypergraph.

        Returns:
            Adj: incidence graph of the input hypergraph.
        """
        offset = torch.Tensor([[0], [self.num_nodes]])
        return (hyperedge_index + offset).to(torch.long)

    def generate_node_features(self, incidence_graph: Adj) -> FeatureTensorType:
        model = Node2Vec(
            incidence_graph, embedding_dim=128, walk_length=5, context_size=1
        )

        return model()

    def generate_hyperedge_index(self) -> [Adj, torch.Tensor, torch.Tensor]:
        """
        Generates a heterogeneous hyperedge index from the incidence matrices.

        The returned hyperedge index is a tensor [V; E] where the first row gives the
        node index and the second row gives the hyperedge index.

        Returns:
            Adj: hyper edge index.
        """

        hyperedge_idx = utils.generate_hyperedge_index(
            incidence_matrices,
            self.edge_type_map,
            self.edge_type_names,
            self.node_type_names,
        )

        self.hyperedges_per_type = hyperedge_idx.hyperedges_per_type
        self.nodes_per_type = hyperedge_idx.nodes_per_type

        return (
            hyperedge_idx.hyperedge_index,
            hyperedge_idx.hyperedge_types,
            hyperedge_idx.node_types,
        )

    def get_incidence_matrices(self):
        incidence_matrices: list[tuple[str, Adj]] = []
        for path in self.raw_file_names:
            filename = Path(path).stem

            incidence_matrix = torch.Tensor(
                np.genfromtxt(f"{self.raw_dir}/{path}")
            ).to_sparse_coo()

            incidence_matrices.append((filename, incidence_matrix))
        return incidence_matrices

    def process(self):
        incidence_matrices = self.get_incidence_matrices()
        hyperedge_idx = utils.generate_hyperedge_index(
            incidence_matrices,
            self.edge_type_map,
            self.edge_type_names,
            self.node_type_names,
        )
        # hyperedge_index, hyperedge_types, node_types = self.generate_hyperedge_index()
        self.nodes_per_type = hyperedge_idx.nodes_per_type
        self.hyperedges_per_type = hyperedge_idx.hyperedges_per_type
        incidence_graph = utils.generate_incidence_graph(
            hyperedge_idx.hyperedge_index, self.num_nodes
        )
        features = utils.generate_node_features(incidence_graph)

        print(features.shape)
        print(self.num_nodes)
        print(self.num_hyperedges)

    def print_summary(self):
        print("======== Dataset summary ========")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of hyperedges: {self.num_hyperedges}")
        print(f"Nodes per type: {self.nodes_per_type}")
        print(f"Hyperedge per type: {dict(self.hyperedges_per_type)}")


if __name__ == "__main__":
    dataset = DTIDataset(root_dir="data", dataset="deepDTnet_20")
    dataset.print_summary()
