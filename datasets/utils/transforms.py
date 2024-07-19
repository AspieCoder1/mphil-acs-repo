#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Union, Optional

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage, EdgeType
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (remove_self_loops, mask_to_index, index_to_mask,
                                   negative_sampling, )


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
        print(torch.unique(store['y'][train_idx_org]))

        if train_idx_org.shape[0] < num_nodes:
            num_val = round(train_idx_org.shape[0] * self.val_ratio)
        else:
            num_val = round(num_nodes * self.val_ratio)

        perm = torch.randperm(train_idx_org.shape[0])
        val_idx = train_idx_org[perm][:num_val]
        train_idx = train_idx_org[perm][num_val:]
        return index_to_mask(train_idx, num_nodes), index_to_mask(val_idx, num_nodes)


class TrainValEdgeSplit(BaseTransform):

    def __init__(self, train_ratio: float = 0.9,
                 target: Optional[EdgeType] = None):
        """
        Transformation to perform splitting of edge index into a train and validation
        set. Adapted from the HGB link prediction splitting code avaliable at:
        https://github.com/THUDM/HGB/blob/master/LP/benchmark/scripts/data_loader.py

        Args:
            train_ratio: ratio of edges used in the training set.
        """
        super(TrainValEdgeSplit).__init__()

        self.train_ratio = train_ratio
        self.edge_type = target

    def forward(self, data: HeteroData) -> HeteroData:
        assert isinstance(data, HeteroData), 'data must be of type HeteroData'

        store = data[self.edge_type]

        train_index_pos, valid_index_pos = self.get_pos_samples(store)

        train_index_neg = negative_sampling(store.edge_index,
                                            num_neg_samples=train_index_pos.shape[1])
        valid_index_neg = negative_sampling(store.edge_index,
                                            num_neg_samples=valid_index_pos.shape[1])

        train_edge_labels = torch.cat(
            [torch.ones_like(train_index_pos[0]), torch.zeros_like(train_index_neg[0])],
            dim=-1)
        val_edge_labels = torch.cat([torch.ones_like(valid_index_pos[0]),
                                     torch.zeros_like(valid_index_neg[0])], dim=-1)

        store['edge_index'] = train_index_pos
        store['train_edge_label_index'] = torch.cat((train_index_pos, train_index_neg),
                                                    dim=-1)
        store['train_edge_label'] = train_edge_labels
        store['val_edge_label_index'] = torch.cat((valid_index_pos, valid_index_neg),
                                                  dim=-1)
        store['val_edge_label'] = val_edge_labels

        data[self.edge_type].update(store)

        return data

    def get_pos_samples(self, store: EdgeStorage) -> [torch.Tensor, torch.Tensor]:
        # get first occurrence of each node as src
        unique, idx, counts = torch.unique(store.edge_index[0], sorted=True,
                                           return_inverse=True,
                                           return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]

        train_edge_index_init = torch.index_select(store.edge_index, dim=1,
                                                   index=first_indicies)

        remaining_mask = torch.ones_like(store.edge_index[0])
        remaining_mask[first_indicies] = 0

        remaining_edge_index = store.edge_index[:, remaining_mask]

        train_mask = torch.randn(remaining_edge_index.shape[1]) < self.train_ratio

        train_edge_index = torch.cat(
            (train_edge_index_init, store.edge_index[:, train_mask]), dim=1)
        val_edge_index = store.edge_index[:, ~train_mask]

        return train_edge_index, val_edge_index
