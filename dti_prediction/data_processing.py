#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os.path as osp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Tuple, Literal, Mapping

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.nn.models import Node2Vec
from torch_geometric.typing import Adj, FeatureTensorType


class PreprocessDataset(ABC):
    def __init__(self): ...

    @abstractmethod
    def generate_hyperedge_index(self):
        raise NotImplementedError

    @abstractmethod
    def generate_incidence_graph(self):
        raise NotImplementedError

    @abstractmethod
    def get_feature_matrix(self):
        raise NotImplementedError


class DTIDatasets(InMemoryDataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        pre_transform=None,
        dataset: Literal["deepDTnet_20", "KEGG_MED", "DTINet_17"] = "deepDTNet_20",
    ):
        super(DTIDatasets, self).__init__(root_dir, transform, pre_transform)
        self.dataset = dataset
        self.edge_type_map = {
            ("drug", "disease"): "drug_treats",
            ("drug", "protein"): "drug_targets",
            ("protein", "drug"): "protein_interacts_with",
            ("protein", "disease"): "protein_linked_to",
            ("disease", "drug"): "disease_treated_by",
            ("disease", "protein"): "disease_linked_to",
        }
        self.edge_type_names = [
            "drug_treats",
            "drug_targets",
            "protein_interacts_with",
            "protein_linked_to",
            "disease_treated_by",
            "disease_linked_to",
        ]
        self.node_type_names = ["drug", "protein", "disease"]
        self.nodes_per_type: Mapping[str, int] = {}
        self.file_ids: Mapping[str, str] = {
            "deepDTnet_20": "1RGS2K58Gjr5IxPJTE4G-MHl0S6Wk6UgZ",
            "KEGG_MED": "1_XOT7Czd560UvkxpJM1-L5t9GXDPLhQr",
            "DTINet_17": "1pLoNyznbcTaxBHW8cSNPUU6oN3WCAh3l",
        }

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["drug_disease.txt", "drug_protein.txt", "protein_disease.txt"]

    @property
    def num_nodes(self):
        return sum(self.nodes_per_type.values())

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root_dir, self.dataset, "raw")

    def download(self):
        url = f"https://drive.google.com/uc?export=download&id={self.file_ids[self.dataset]}"
        path = download_url(url=url, folder=self.raw_dir, filename="data.zip")
        extract_zip(self.raw_dir, path)

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
        return hyperedge_index + offset

    def generate_node_features(self, incidence_graph: Adj) -> FeatureTensorType:
        model = Node2Vec(
            incidence_graph, embedding_dim=128, walk_length=5, context_size=10
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
        current_node_idx = 0
        current_hyperedge_idx = 0
        node_idx_start: Mapping[str, int] = {}
        hyperedge_idxs: list[Adj] = []
        edge_types: list[torch.Tensor] = []
        for path in self.raw_paths:
            filename = Path(path).stem
            src, dst = filename.split("_")
            incidence_matrix = torch.Tensor(
                np.genfromtxt(f"{self.raw_dir}/{path}")
            ).to_sparse()

            # Handle initial direction
            hyperedge_idx = incidence_matrix.indices()

            if src not in node_idx_start:
                node_idx_start[src] = current_node_idx
                current_node_idx += incidence_matrix.shape[0]
                self.nodes_per_type[src] = incidence_matrix.shape[0]
            offset = torch.Tensor([[node_idx_start[src]], [current_hyperedge_idx]]).to(
                torch.long
            )
            hyperedge_idx += offset
            current_hyperedge_idx += hyperedge_idx.shape[1]
            hyperedge_type = self.edge_type_names.index(self.edge_type_map[(src, dst)])
            hyperedge_types = hyperedge_type * torch.ones(
                torch.unique(hyperedge_idx[1]).shape[0]
            )

            # Handle transpose
            hyperedge_idx_inverse = incidence_matrix.T.coalesce().indices()
            if dst not in node_idx_start:
                node_idx_start[dst] = current_node_idx
                current_node_idx += incidence_matrix.shape[1]
                self.nodes_per_type[dst] = incidence_matrix.shape[1]
            offset = torch.Tensor([[node_idx_start[src]], [current_hyperedge_idx]]).to(
                torch.long
            )
            hyperedge_idx_inverse += offset
            current_hyperedge_idx += hyperedge_idx_inverse.shape[1]
            inverse_hyperedge_type = self.edge_type_names.index(
                self.edge_type_map[(dst, src)]
            )
            inverse_hyperedge_types = inverse_hyperedge_type * torch.ones(
                torch.unique(hyperedge_idx[1]).shape[0]
            )

            hyperedge_idxs.extend([hyperedge_idx, hyperedge_idx_inverse])
            edge_types.extend([hyperedge_types, inverse_hyperedge_types])

        node_types = []
        for k, v in self.nodes_per_type.items():
            node_types.append(self.node_type_names.index(k) * torch.ones(v))

        hyperedge_index = torch.cat(hyperedge_idxs, dim=1)
        hyperedge_types = torch.cat(edge_types, dim=0)
        node_types = torch.cat(node_types, dim=0)

        return hyperedge_index, hyperedge_types, node_types

    def process(self):
        hyperedge_index, hyperedge_types, node_types = self.generate_hyperedge_index()
        incidence_graph = self.generate_incidence_graph(hyperedge_index)
        features = self.generate_node_features(incidence_graph)
        node_features = features[:, :self.num_nodes]
        hyeredge_features = features[:, self.num_nodes:]
