#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Union, Optional

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.mask import mask_to_index, index_to_mask
from torch_geometric.data.storage import NodeStorage
from torch_geometric.utils import remove_self_loops


class RemoveSelfLoops(BaseTransform):
    def __init__(self):
        self.attr = "edge_weight"
        super().__init__()

    def forward(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store[self.attr] = remove_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
            )

        return data


class TrainValNodeSplit(BaseTransform):
    def __init__(self, val_ratio: float = 0.2, key: Optional[str] = "y"):
        super().__init__()
        self.val_ratio = val_ratio
        self.key = key

    def forward(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_mask, val_mask = self._split(store)
            store.train_mask = train_mask
            store.val_mask = val_mask
        return data

    def _split(self, store: NodeStorage):
        num_nodes = store.num_nodes
        assert num_nodes is not None

        train_idx_org = mask_to_index(store['train_mask'])

        num_val = round(num_nodes * self.val_ratio)
        perm = torch.randperm(train_idx_org.shape[0])

        train_idx = train_idx_org[perm][:num_val]
        val_idx = train_idx_org[perm][num_val:]

        return index_to_mask(train_idx, num_nodes), index_to_mask(val_idx, num_nodes)


