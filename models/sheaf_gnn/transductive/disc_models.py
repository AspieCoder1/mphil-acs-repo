#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion
from abc import abstractmethod

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn

from models.sheaf_gnn import laplacian_builders as lb
from models.sheaf_gnn.orthogonal import Orthogonal
from models.sheaf_gnn.sheaf_base import SheafDiffusion
from models.sheaf_gnn.sheaf_models import (
    EdgeWeightLearner,
    LocalConcatSheafLearnerVariant,
)
from ..utils import init_sheaf_learner


class DiscreteSheafDiffusion(SheafDiffusion):
    def __init__(
        self,
        edge_index,
        args,
        sheaf_learner: str = "local_concat",
    ):
        super(DiscreteSheafDiffusion, self).__init__(edge_index, args)
        self.sheaf_learner = init_sheaf_learner(sheaf_learner)
        self.sheaf_type = sheaf_learner

    @abstractmethod
    def process_restriction_maps(self, maps): ...

    @abstractmethod
    def forward(self, x: torch.Tensor, node_types: torch.Tensor,
                edge_types: torch.Tensor): ...


class DiscreteDiagSheafDiffusion(DiscreteSheafDiffusion):

    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteDiagSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=True)
                )
                nn.init.xavier_normal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_feat = nn.Dropout(self.dropout)
        self.dropout_input = nn.Dropout(self.input_dropout)

    def forward(self, x: torch.Tensor, node_types, edge_types):
        x = self.dropout_input(x)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x

        embs = []
        for layer in range(self.layers):
            x = self.dropout_feat(x)
            if layer == 0 or self.nonlinear:
                x_maps = x

                # maps are the linear restriction maps
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.graph_size, -1),
                    self.edge_index,
                    edge_types,
                    node_types,
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

                x0 = x0.t().reshape(-1, self.final_d)
                x0 = self.lin_left_weights[layer](x0)
                x0 = x0.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)
                x0 = self.lin_right_weights[layer](x0)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # if self.use_act:
            #     x = F.elu(x)
            x0 = F.elu(x)
            x = x0

            if self.use_hidden_embeddings:
                embs.append(F.normalize(x, p=2))

        if self.use_hidden_embeddings:
            x = torch.hstack(embs)

        x = x.reshape(self.graph_size, -1)

        if not self.use_hidden_embeddings:
            return F.normalize(x, p=2, dim=1)
        # x = self.lin2(x)
        return x

    def process_restriction_maps(self, maps):
        return maps

    def __str__(self):
        return f"DiagSheaf-{self.sheaf_type}"


class DiscreteBundleSheafDiffusion(DiscreteSheafDiffusion):

    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteBundleSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.xavier_normal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )

            if self.use_edge_weights:
                self.weight_learners.append(
                    EdgeWeightLearner(self.hidden_dim, edge_index)
                )
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.dropout_input = nn.Dropout(self.input_dropout)

    def get_param_size(self):
        if self.orth_trans in ["matrix_exp", "cayley"]:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x: torch.Tensor, node_types, edge_types):
        x = self.dropout_input(x)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        embs = []
        for layer in range(self.layers):
            x = self.dropout_layer(x)
            if layer == 0 or self.nonlinear:
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](
                    x_maps, self.edge_index, edge_types, node_types
                )
                edge_weights = (
                    self.weight_learners[layer](x_maps, self.edge_index)
                    if self.use_edge_weights
                    else None
                )
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )
            x0 = self.left_right_linear(x0, self.lin_left_weights[layer],
                                        self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            x0 = F.elu(x)
            x = x0

            if self.use_hidden_embeddings:
                embs.append(F.normalize(x, p=2))

        if self.use_hidden_embeddings:
            x = torch.hstack(embs)

        x = x.reshape(self.graph_size, -1)

        if not self.use_hidden_embeddings:
            return F.normalize(x, p=2, dim=1)
        # x = self.lin2(x)
        return x

    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor:
        transform = Orthogonal(self.d, self.orth_trans)
        maps = transform(maps)
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        return f"BundleSheaf-{self.sheaf_type}"


class DiscreteGeneralSheafDiffusion(DiscreteSheafDiffusion):

    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteGeneralSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.xavier_normal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_lp=self.add_lp,
            add_hp=self.add_hp,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
        )

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.dropout_input = nn.Dropout(self.input_dropout)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x: torch.Tensor, node_types, edge_types):
        x = self.dropout_input(x)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        embs = []
        for layer in range(self.layers):
            x = self.dropout_layer(x)
            if layer == 0 or self.nonlinear:
                maps = self.sheaf_learners[layer](
                    x.reshape(self.graph_size, -1),
                    self.edge_index,
                    edge_types,
                    node_types,
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )
            x0 = self.left_right_linear(
                x0, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )


            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            x0 = F.elu(x0 - x)
            x = x0

            if self.use_hidden_embeddings:
                embs.append(F.normalize(x, p=2))

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        if self.use_hidden_embeddings:
            x = torch.hstack(embs)

        x = x.reshape(self.graph_size, -1)

        if not self.use_hidden_embeddings:
            return F.normalize(x, p=2, dim=1)
        # x = self.lin2(x)
        return x

    def process_restriction_maps(self, maps):
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        return f"GeneralSheaf-{self.sheaf_type}"
