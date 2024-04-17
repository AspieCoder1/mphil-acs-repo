#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.typing import Adj


@dataclass(slots=True)
class HyperedgeIndex:
    hyperedge_index: Adj
    hyperedge_types: torch.Tensor
    node_types: torch.Tensor
    hyperedges_per_type: dict[str, int]
    nodes_per_type: dict[str, int]


def generate_hyperedge_index(
    incidence_matrics: list[tuple[str, Adj]],
    edge_type_map,
    edge_type_names,
    node_type_names,
):
    node_idx_start = {}
    nodes_per_type = {}
    hyperedges_per_type = defaultdict(lambda: 0)
    current_node_idx = 0
    current_hyperedge_idx = 0
    edge_types = []
    hyperedge_idxs = []

    for filename, incidence_matrix in incidence_matrics:
        src, dst = filename.split("_")  # data types stored in rows and cols

        # Rows are nodes and hyperedges are columns
        hyperedge_idx = incidence_matrix.to_sparse_coo().indices()
        if src not in node_idx_start.keys():
            node_idx_start[src] = current_node_idx
            nodes_per_type[src] = incidence_matrix.shape[0]
            current_node_idx += incidence_matrix.shape[0]

        offset = torch.Tensor([[node_idx_start[src]], [current_hyperedge_idx]]).to(
            torch.long
        )
        hyperedge_idx += offset
        current_hyperedge_idx += incidence_matrix.shape[1]
        hyperedge_type = edge_type_names.index(edge_type_map[(src, dst)])
        hyperedge_types = hyperedge_type * torch.ones(incidence_matrix.shape[1])
        hyperedges_per_type[edge_type_names[hyperedge_type]] += incidence_matrix.shape[
            1
        ]

        # Rows are hyperedges and columns are nodes
        hyperedge_idx_inverse = (
            incidence_matrix.T.to_sparse_coo().coalesce().indices()
        )

        if dst not in node_idx_start.keys():
            node_idx_start[dst] = current_node_idx
            nodes_per_type[dst] = incidence_matrix.shape[1]
            current_node_idx += incidence_matrix.shape[1]

        offset = torch.Tensor([[node_idx_start[dst]], [current_hyperedge_idx]]).to(
            torch.long
        )
        hyperedge_idx_inverse += offset
        current_hyperedge_idx += incidence_matrix.shape[0]
        inverse_hyperedge_type = edge_type_names.index(edge_type_map[(dst, src)])
        inverse_hyperedge_types = inverse_hyperedge_type * torch.ones(
            incidence_matrix.shape[0]
        )
        hyperedges_per_type[
            edge_type_names[inverse_hyperedge_type]
        ] += incidence_matrix.shape[0]

        hyperedge_idxs.extend([hyperedge_idx, hyperedge_idx_inverse])
        edge_types.extend([hyperedge_types, inverse_hyperedge_types])

    node_types = []
    for k, v in nodes_per_type.items():
        node_types.append(node_type_names.index(k) * torch.ones(v))

    hyperedge_index = torch.column_stack(hyperedge_idxs)
    hyperedge_types = torch.cat(edge_types, dim=0)
    node_types = torch.cat(node_types, dim=0)

    return HyperedgeIndex(
        hyperedge_index=hyperedge_index,
        hyperedge_types=hyperedge_types,
        node_types=node_types,
        hyperedges_per_type=dict(hyperedges_per_type),
        nodes_per_type=nodes_per_type,
    )


def generate_incidence_graph(hyperedge_index: Adj):
    max_node_idx = torch.max(hyperedge_index[0]).item() + 1
    offset = torch.Tensor([[0], [max_node_idx]])
    return (hyperedge_index + offset).to(torch.long)


def generate_node_features(incidence_graph: Adj):
    model = Node2Vec(incidence_graph, embedding_dim=128, walk_length=5, context_size=1)

    return model()
