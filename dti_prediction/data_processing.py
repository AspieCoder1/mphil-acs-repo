#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os.path as osp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Tuple, Literal

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.typing import Adj


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

    @property
    def edge_type_map(self):
        return {
            ("drug", "disease"): "drug_treats",
            ("drug", "protein"): "drug_interacts_with",
            ("protein", "drug"): "protein_interacts_with",
            ("protein", "disease"): "protein_linked_to",
            ("disease", "drug"): "disease_treated_by",
            ("disease", "protein"): "disease_linked_to",
        }

    @property
    def edge_type_names(self):
        return [
            "drug_treats",
            "drug_targets",
            "protein_interacts_with",
            "protein_linked_to",
            "disease_treated_by",
            "disease_linked_to",
        ]

    @property
    def node_type_names(self):
        return ["drug", "protein", "disease"]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["drug_disease.txt", "drug_protein.txt", "protein_disease.txt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root_dir, self.dataset, "raw")

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
        max_node_idx, _ = torch.max(hyperedge_index, dim=1, keepdim=True)
        max_node_idx[0] = 0

        return hyperedge_index + max_node_idx

    def generate_hyperedge_index(self) -> Tuple[Adj, torch.Tensor, torch.Tensor]:
        """
        Generates a heterogeneous hyperedge index from the incidence matrices.

        The returned hyperedge index is a tensor [V; E] where the first row gives the
        node index and the second row gives the hyperedge index.

        Returns:
            Adj: hyper edge index.
        """
        hyperedge_idxs = []
        node_types = []
        edge_types = []
        for path in self.raw_paths:
            # Handle the standard incidence matrix
            max_idx = 0
            if len(hyperedge_idxs) > 0:
                max_idx, _ = torch.max(hyperedge_idxs[-1], dim=1, keepdim=True)
                max_idx += 1

            H = torch.Tensor(np.genfromtxt(f"{path}")).to_sparse()

            filename = Path(path).stem
            print(filename)

            src, dst = filename.split("_")
            hyperedge_idx = H.indices()
            hyperedge_idx += max_idx

            max_idx, _ = torch.max(hyperedge_idx, dim=1, keepdim=True)
            max_idx += 1
            hyperedge_idx_T = H.T.coalesce().indices() + max_idx

            hyperedge_idxs.extend([hyperedge_idx, hyperedge_idx_T])

            edge_type = self.edge_type_names.index(self.edge_type_map[(src, dst)])
            node_type = self.node_type_names.index(src)
            edge_type_T = self.edge_type_names.index(self.edge_type_map[(dst, src)])
            node_type_T = self.node_type_names.index(dst)

            types = torch.ones((hyperedge_idx.shape[0], 1))

            edge_types.extend([edge_type * types, edge_type_T * types])
            node_types.extend([node_type * types, node_type_T * types])

        hyperedge_index = torch.cat(hyperedge_idxs, dim=1)
        node_types = torch.cat(node_types, dim=1)
        hyperedge_types = torch.cat(edge_types, dim=1)

        return hyperedge_index, node_types, hyperedge_types


if __name__ == "__main__":
    print("Test")
