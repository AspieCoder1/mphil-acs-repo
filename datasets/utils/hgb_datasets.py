#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)
from torch_geometric.utils import to_edge_index
from torch_geometric.transforms.to_undirected import ToUndirected

from .hgb_loaders import HGBDataLoaderLP


class HGBDatasetNC(InMemoryDataset):
    r"""A variety of heterogeneous graph benchmark datasets from the
    `"Are We Really Making Much Progress? Revisiting, Benchmarking, and
    Refining Heterogeneous Graph Neural Networks"
    <http://keg.cs.tsinghua.edu.cn/jietang/publications/
    KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.

    .. note::
        Test labels are randomly given to prevent data leakage issues.
        If you want to obtain final test performance, you will need to submit
        your model predictions to the
        `HGB leaderboard <https://www.biendata.xyz/hgb/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"ACM"`,
            :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
        transform (callable, optional): A function/transform that takes in an
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    names = {
        'acm': 'ACM',
        'dblp': 'DBLP',
        'freebase': 'Freebase',
        'imdb': 'IMDB',
        'pubmed_nc': 'PubMed_NC'
    }

    file_ids = {
        'acm': '1xbJ4QE9pcDJOcALv7dYhHDCPITX2Iddz',
        'dblp': '1fLLoy559V7jJaQ_9mQEsC06VKd6Qd3SC',
        'freebase': '1vw-uqbroJZfFsWpriC1CWbtHCJMGdWJ7',
        'imdb': '18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2',
        'pubmed_nc': '18symt1BUf4d6Gge_uapt3B0rh9ohux7c',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.names
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        id = self.file_ids[self.name]
        path = download_google_url(id, self.raw_dir, 'data.zip')
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        data = HeteroData()

        # node_types = {0: 'paper', 1, 'author', ...}
        # edge_types = {0: ('paper', 'cite', 'paper'), ...}
        if self.name in ['acm', 'dblp', 'imdb', 'pubmed_nc']:
            with open(self.raw_paths[0]) as f:  # `info.dat`
                info = json.load(f)
            n_types = info['node.dat']['node type']
            n_types = {int(k): v for k, v in n_types.items()}
            e_types = info['link.dat']['link type']
            e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
            for key, (src, dst, rel) in e_types.items():
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                rel = rel if rel != dst and rel[1:] != dst else 'to'
                e_types[key] = (src, rel, dst)
            num_classes = len(info['label.dat']['node type']['0'])
        elif self.name in ['freebase']:
            with open(self.raw_paths[0]) as f:  # `info.dat`
                info = f.read().split('\n')
            start = info.index('TYPE\tMEANING') + 1
            end = info[start:].index('')
            n_types = [v.split('\t\t') for v in info[start:start + end]]
            n_types = {int(k): v.lower() for k, v in n_types}

            e_types = {}
            start = info.index('LINK\tSTART\tEND\tMEANING') + 1
            end = info[start:].index('')
            for key, row in enumerate(info[start:start + end]):
                row = row.split('\t')[1:]
                src, dst, rel = (v for v in row if v != '')
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                e_types[key] = (src, rel, dst)
        else:  # Link prediction:
            raise NotImplementedError

        # Extract node information:
        mapping_dict = {}  # Maps global node indices to local ones.
        x_dict = defaultdict(list)
        num_nodes_dict: Dict[str, int] = defaultdict(int)
        with open(self.raw_paths[1]) as f:  # `node.dat`
            xs = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for x in xs:
            n_id, n_type = int(x[0]), n_types[int(x[2])]
            mapping_dict[n_id] = num_nodes_dict[n_type]
            num_nodes_dict[n_type] += 1
            if len(x) >= 4:  # Extract features (in case they are given).
                x_dict[n_type].append([float(v) for v in x[3].split(',')])
        for n_type in n_types.values():
            if len(x_dict[n_type]) == 0:
                data[n_type].num_nodes = num_nodes_dict[n_type]
            else:
                data[n_type].x = torch.tensor(x_dict[n_type])

        edge_index_dict = defaultdict(list)
        edge_weight_dict = defaultdict(list)
        with open(self.raw_paths[2]) as f:  # `link.dat`
            edges = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for src, dst, rel, weight in edges:
            e_type = e_types[int(rel)]
            src, dst = mapping_dict[int(src)], mapping_dict[int(dst)]
            edge_index_dict[e_type].append([src, dst])
            edge_weight_dict[e_type].append(float(weight))
        for e_type in e_types.values():
            edge_index = torch.tensor(edge_index_dict[e_type])
            edge_weight = torch.tensor(edge_weight_dict[e_type])
            data[e_type].edge_index = edge_index.t().contiguous()
            # Only add "weighted" edgel to the graph:
            if not torch.allclose(edge_weight, torch.ones_like(edge_weight)):
                data[e_type].edge_weight = edge_weight

        # Node classification:
        if self.name in ['acm', 'dblp', 'freebase', 'imdb', 'pubmed_nc']:
            with open(self.raw_paths[3]) as f:  # `label.dat`
                train_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            with open(self.raw_paths[4]) as f:  # `label.dat.test`
                test_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
            for y in train_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]

                if not hasattr(data[n_type], 'y'):
                    num_nodes = data[n_type].num_nodes
                    if self.name in ['imdb']:  # multi-label
                        data[n_type].y = torch.zeros((num_nodes, num_classes))
                    else:
                        data[n_type].y = torch.full((num_nodes,), -1).long()
                    data[n_type].train_mask = torch.zeros(num_nodes).bool()
                    data[n_type].test_mask = torch.zeros(num_nodes).bool()

                if data[n_type].y.dim() > 1:  # multi-label
                    for v in y[3].split(','):
                        data[n_type].y[n_id, int(v)] = 1
                else:
                    data[n_type].y[n_id] = int(y[3])
                data[n_type].train_mask[n_id] = True
            for y in test_ys:
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]
                if data[n_type].y.dim() > 1:  # multi-label
                    for v in y[3].split(','):
                        data[n_type].y[n_id, int(v)] = 1
                else:
                    data[n_type].y[n_id] = int(y[3])
                data[n_type].test_mask[n_id] = True

        else:  # Link prediction:
            raise NotImplementedError

        transform = ToUndirected(merge=False)
        data = transform(data)

        num_nodes = torch.tensor([data[node_type].num_nodes for node_type in data.node_types])
        num_edges = torch.tensor([data[edge_type].edge_index.shape[1] for edge_type in data.edge_types])
        node_types = torch.arange(len(data.node_types)).repeat_interleave(num_nodes)
        edge_types = torch.arange(len(data.edge_types)).repeat_interleave(num_edges)

        data.node_type = node_types
        data.edge_type = edge_types

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def get_metadata(self):
        with open(self.raw_paths[0]) as f:  # `info.dat`
            info = json.load(f)
        if self.name == 'lastfm':
            n_types = info['node.dat']
            e_types = info['link.dat']
        else:
            n_types = info['node.dat']['node type']
            e_types = info['link.dat']['link type']
        n_types = {int(k): v for k, v in n_types.items()}
        e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
        for key, (src, dst, rel) in e_types.items():
            src, dst = n_types[int(src)], n_types[int(dst)]
            rel = rel.split('-')[1]
            rel = rel if rel != dst and rel[1:] != dst else 'to'
            e_types[key] = (src, rel, dst)

        return e_types, n_types

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'


class HGBDatasetLP(InMemoryDataset):
    names = {
        "lastfm": "LastFM",
        "pubmed_lp": "PubMed_LP",
    }
    file_ids = {
        "lastfm": "1busKxUoPOWZa7xJIV0kgPfteDn6bpYKK",
        "pubmed_lp": "1syDE6wacF6f3XezcU66RiTlRGABI6ktC",
    }

    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
            n_splits = 5
    ) -> None:
        self.name = name.lower()
        assert self.name in set(self.names.keys())
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
        self.n_splits = n_splits
        self.dl: Optional[HGBDataLoaderLP] = None

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'link.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        id = self.file_ids[self.name]
        path = download_google_url(id, self.raw_dir, 'data.zip')
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = HeteroData()

        # 1. Get correct data loader from dataset
        self.dl = HGBDataLoaderLP(path=osp.join(self.raw_dir, self.names[self.name]))

        # 2. get correct metadata object
        e_types, n_types, n_types_inv = self.get_metadata()

        # 3. generate node features
        for index, node_type in n_types.items():
            if self.dl.nodes['attr'][index] is None:
                data[node_type].x = self.dl.nodes['attr'][index]
            else:
                data[node_type].x = torch.tensor(self.dl.nodes['attr'][index]).to(torch.float)
            data[node_type].num_nodes = self.dl.nodes['count'][index]

        # 4. add edge indices
        for index, e_type in e_types.items():
            src_type, dst_type = self.dl.links["meta"][index]
            csr = self.dl.links['data'][index]

            sparse_adj = torch.sparse_coo_tensor(np.array(csr.nonzero()), csr.data,
                                                 csr.shape)
            edge_index, _ = to_edge_index(sparse_adj)
            offset = torch.tensor(
                [[self.dl.nodes['shift'][src_type]], [self.dl.nodes['shift'][dst_type]]])

            edge_index -= offset
            data[e_type].edge_index = edge_index

        target = self.dl.test_types[0]

        # 4. add test samples
        test_neigh, test_labels = self.dl.get_test_neigh()
        src, _, dst = e_types[target]
        src, dst = n_types_inv[src], n_types_inv[dst]
        offset = torch.tensor(
            [[self.dl.nodes['shift'][src]], [self.dl.nodes['shift'][dst]]])
        test_edge_label_index = torch.tensor(test_neigh[target]) - offset
        test_edge_label = torch.tensor(test_labels[target])
        data[e_types[target]].test_edge_label_index = test_edge_label_index
        data[e_types[target]].test_edge_label = test_edge_label

        # Adding the customised type information
        transform = ToUndirected(merge=False)
        data = transform(data)

        num_nodes = torch.tensor(
            [data[node_type].num_nodes for node_type in data.node_types])
        num_edges = torch.tensor(
            [data[edge_type].edge_index.shape[1] for edge_type in data.edge_types])
        node_types = torch.arange(len(data.node_types)).repeat_interleave(num_nodes)
        edge_types = torch.arange(len(data.edge_types)).repeat_interleave(num_edges)

        data.node_type = node_types
        data.edge_type = edge_types

        self.save([data], self.processed_paths[0])

    def get_metadata(self):
        with open(self.raw_paths[0]) as f:  # `info.dat`
            info = json.load(f)
        if self.name == 'lastfm':
            n_types = info['node.dat']
            e_types = info['link.dat']
        else:
            n_types = info['node.dat']['node type']
            e_types = info['link.dat']['link type']
        n_types = {int(k): v for k, v in n_types.items()}
        n_types_inv = {v: int(k) for k, v in n_types.items()}
        e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
        for key, (src, dst, rel) in e_types.items():
            src, dst = n_types[int(src)], n_types[int(dst)]
            rel = rel.split('-')[1]
            rel = rel if rel != dst and rel[1:] != dst else 'to'
            e_types[key] = (src, rel, dst)

        return e_types, n_types, n_types_inv
