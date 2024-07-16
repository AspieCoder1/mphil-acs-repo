#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from abc import abstractmethod
from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Adj, InputNodes, OptTensor

from .lib import laplace as lap


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""

    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(
        self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh", **kwargs
    ):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2, int(np.prod(out_shape)), bias=False
        )

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.linear1(torch.cat([x_src, x_dst], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "local_concat"


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(
        self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"
    ):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(
            hidden_channels * 2, int(np.prod(out_shape)), bias=False
        )
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index

        x_src = torch.index_select(x, dim=0, index=src)  # this is really x_src
        x_dst = torch.index_select(x, dim=0, index=dst)  # this is really x_dst
        x_cat = torch.cat([x_src, x_dst], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "local_concat"


class AttentionSheafLearner(SheafLearner):

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels * 2, d**2, bias=False)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.linear1(torch.cat([x_src, x_dst], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels * 2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_src = torch.index_select(x, dim=0, index=row)
        x_dst = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_src, x_dst], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(
            weights, index=full_right_idx, dim=0
        )
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.map_builder(torch.cat([x_src, x_dst], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])


class TypeConcatSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(TypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_node_types * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types)

        x_cat = torch.cat(
            [x_src, x_dst, src_type, dst_type, edge_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "type_concat"


class TypeEnsembleSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(TypeEnsembleSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_node_types * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.linear1 = nn.ModuleList(
            [
                nn.Linear(in_channels * 2, int(np.prod(out_shape)), bias=False)
                for _ in range(num_edge_types)
            ]
        )

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def compute_map(self, x_cat: torch.Tensor, edge_type):
        return self.linear1[edge_type](x_cat)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        x_cat = torch.cat(
            [x_src, x_dst],
            dim=1,
        )

        unique, counts = torch.unique(edge_types, return_counts=True)
        edge_type_idx = torch.argsort(edge_types)
        edge_type_splits = edge_type_idx.split(split_size=counts.tolist())

        results = []

        for i, split in enumerate(edge_type_splits):
            results.append(self.linear1[i](x_cat[split]))

        stacked_maps = torch.row_stack(results)

        maps = torch.empty(stacked_maps.shape, device=stacked_maps.device)
        maps[edge_type_idx] = stacked_maps
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "type_ensemble"


class EdgeTypeConcatSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(EdgeTypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types)

        x_cat = torch.cat(
            [x_src, x_dst, edge_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "edge_type_concat"


class NodeTypeConcatSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(NodeTypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_node_types * 2,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)

        x_cat = torch.cat(
            [x_src, x_dst, src_type, dst_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "node_type_concat"


class NodeTypeSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(NodeTypeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            num_node_types * 2,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)

        x_cat = torch.cat(
            [src_type, dst_type],
            dim=1,
        ).to(torch.float)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "node_type"


class EdgeTypeSheafLearner(SheafLearner):
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(EdgeTypeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types).to(
            torch.float
        )

        maps = self.linear1(edge_type)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "edge_type"


class TypeSheafLearner(SheafLearner):
    def __init__(
            self,
            in_channels: int,
            out_shape: Tuple[int, ...],
            sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
            num_node_types: int = 4,
            num_edge_types: int = 12,
    ):
        super(TypeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            2 * num_node_types + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
            self,
            x: InputNodes,
            edge_index: Adj,
            edge_types: OptTensor = None,
            node_types: OptTensor = None,
    ):
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types).to(
            torch.float
        )
        src, dst = edge_index
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)

        x_cat = torch.cat(
            [src_type, dst_type, edge_type],
            dim=1,
        ).to(torch.float)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])
