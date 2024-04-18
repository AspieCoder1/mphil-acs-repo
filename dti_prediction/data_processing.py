#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os.path as osp
from pathlib import Path
from typing import Union, List, Tuple, Literal, Optional, Callable

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.typing import Adj

try:
    from . import utils
except ImportError:
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
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        dataset: Literal["deepDTnet_20", "KEGG_MED", "DTINet_17"] = "deepDTNet_20",
        split: int = 0,
    ):
        """
        DTI dataset for use on Hypergraph Neural Networks.

        Args:
            root_dir: root location to save dataset.
            transform: transformation applied to data
            pre_transform: initial transformations applied before preprocessing
            pre_filter: filtering to apply before processing
            force_reload: reprocess data if already processed
            dataset: dataset to load in
            split: split to use must be between 0 and 9 (inclusive).
        """
        self.dataset = dataset
        self.edge_type_map = EDGE_TYPE_MAP
        self.edge_type_names = EDGE_TYPE_NAMES
        self.node_type_names = NODE_TYPE_NAMES
        self.nodes_per_type: dict[str, int] = {}
        self.hyperedges_per_type: dict[str, int] = {}
        self.split = split
        self.file_ids: dict[str, str] = {
            "deepDTnet_20": "1RGS2K58Gjr5IxPJTE4G-MHl0S6Wk6UgZ",
            "KEGG_MED": "1_XOT7Czd560UvkxpJM1-L5t9GXDPLhQr",
            "DTINet_17": "1pLoNyznbcTaxBHW8cSNPUU6oN3WCAh3l",
        }
        super().__init__(
            root_dir, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        path = osp.join(self.processed_dir, f"data_{self.split}.pt")
        self.load(path)

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
        return f"data_{self.split}.pt"

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.dataset, "processed")

    def download(self):
        url = f"https://drive.google.com/uc?export=download&id={self.file_ids[self.dataset]}"
        path = download_url(url=url, folder=self.raw_dir, filename="data.zip")
        extract_zip(path, self.raw_dir)

    def get_incidence_matrices(self):
        incidence_matrices: list[tuple[str, Adj]] = []
        for path in self.raw_file_names:
            filename = Path(path).stem
            incidence_matrix = torch.Tensor(np.genfromtxt(f"{self.raw_dir}/{path}"))
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
        incidence_graph = utils.generate_incidence_graph(hyperedge_idx.hyperedge_index)
        features = (
            utils.generate_node_features(incidence_graph).detach().to(torch.float)
        )

        max_node_idx = torch.max(hyperedge_idx.hyperedge_index[0]).item() + 1
        node_features = features[:max_node_idx]
        hyperedge_features = features[max_node_idx:]

        idx_offset = torch.Tensor(
            [
                [hyperedge_idx.node_start_idx["drug"]],
                [hyperedge_idx.node_start_idx["protein"]],
            ]
        )
        train_idx = (
            torch.Tensor(np.genfromtxt(f"{self.raw_dir}/train_{self.split}.txt")).T
            + idx_offset
        )
        test_idx = (
            torch.Tensor(np.genfromtxt(f"{self.raw_dir}/test_{self.split}.txt")).T
            + idx_offset
        )

        data = Data(
            x=node_features,
            edge_index=hyperedge_idx.hyperedge_index,
        )
        data.hyperedge_attr = hyperedge_features
        data.node_types = hyperedge_idx.node_types
        data.hyperedge_types = hyperedge_idx.hyperedge_types
        data.num_hyperedges = self.num_hyperedges
        data.n_x = self.num_nodes
        data.train_idx = train_idx
        data.test_idx = test_idx

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], osp.join(self.processed_dir, f"data_{self.split}.pt"))

    def print_summary(self):
        print("======== Dataset summary ========")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of hyperedges: {self.num_hyperedges}")
        print(f"Nodes per type: {self.nodes_per_type}")
        print(f"Hyperedge per type: {dict(self.hyperedges_per_type)}")


if __name__ == "__main__":
    dataset = DTIDataset(
        root_dir="data", dataset="deepDTnet_20", force_reload=True, split=0
    )
    print(dataset[0])
    print(dataset[0].x)
    print(dataset[0].num_nodes)
