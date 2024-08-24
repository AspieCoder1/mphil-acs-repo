#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021
#
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import torch
import torch_sparse
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add

from . import utils
from models.sheaf_gnn.mlp import MLP


def normalisation_matrices(
    x, hyperedge_index, alpha, num_nodes, num_edges, d, norm_type="degree_norm"
):
    # this will return either D^-1/B^-1 or D^(-1/2)/B^-1
    if norm_type == "degree_norm":
        # return D_inv and B_inv used to normalised the laplacian (/propagation)
        # normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[0],
            dim=0,
            dim_size=num_nodes * d,
        )
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges * d,
        )
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == "sym_degree_norm":
        # normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[0],
            dim=0,
            dim_size=num_nodes * d,
        )
        D = D ** (-0.5)
        D[D == float("inf")] = 0

        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges * d,
        )
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == "block_norm":
        # normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(
            alpha * alpha, hyperedge_index[0], dim=0, dim_size=num_nodes * d
        )
        D = 1.0 / D  # can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges * d,
        )
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == "sym_block_norm":
        # normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(
            alpha * alpha, hyperedge_index[0], dim=0, dim_size=num_nodes * d
        )
        D = D ** (-0.5)  # can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges * d,
        )
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B


class HyperDiffusionDiagSheafConv(MessagePassing):
    """
    One layer of Sheaf Diffusion with diagonal Laplacian Y = (I-D^-1/2LD^-1) with L
    normalised with B^-1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        d,
        device,
        dropout=0,
        bias=True,
        norm_type="degree_norm",
        left_proj=None,
        norm=None,
        residual=False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type = norm_type
        self.left_proj = left_proj
        self.norm = norm
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(
                in_channels=d,
                hidden_channels=d,
                out_channels=d,
                num_layers=1,
                dropout=dropout,
                normalisation="ln",
                input_norm=self.norm,
            )

        self.lin = MLP(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            num_layers=1,
            dropout=dropout,
            normalisation="ln",
            input_norm=self.norm,
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.device = device

        self.I_mask = None
        self.Id = None

        self.reset_parameters()

    # to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()

        zeros(self.bias)

    def forward(
        self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1, num_nodes * self.d).t()
        x = self.lin(x)
        data_x = x

        # depending on norm_type D^-1 or D^-1/2
        D_inv, B_inv = normalisation_matrices(
            x, hyperedge_index, alpha, num_nodes, num_edges, self.d, self.norm_type
        )

        if self.norm_type in ["sym_degree_norm", "sym_block_norm"]:
            # compute D^(-1/2) @ X
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(
            hyperedge_index, alpha, size=(num_nodes * self.d, num_edges * self.d)
        )
        H_t = torch.sparse.FloatTensor(
            hyperedge_index.flip([0]),
            alpha,
            size=(num_edges * self.d, num_nodes * self.d),
        )

        # this is because spdiags does not support gpu
        B_inv = utils.sparse_diagonal(
            B_inv, shape=(num_edges * self.d, num_edges * self.d)
        )
        D_inv = utils.sparse_diagonal(
            D_inv, shape=(num_nodes * self.d, num_nodes * self.d)
        )

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(
            B_inv.indices(),
            B_inv.values(),
            H_t.indices(),
            H_t.values(),
            B_inv.shape[0],
            B_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            H.indices(),
            H.values(),
            minus_L[0],
            minus_L[1],
            H.shape[0],
            H.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            D_inv.indices(),
            D_inv.values(),
            minus_L[0],
            minus_L[1],
            D_inv.shape[0],
            D_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch.sparse_coo_tensor(
            minus_L[0], minus_L[1], size=(num_nodes * self.d, num_nodes * self.d)
        ).to(self.device)

        # negate the diagonal blocks and add eye matrix
        if self.I_mask is None:  # prepare these in advance
            I_mask_indices = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0
            )
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse_coo_tensor(I_mask_indices, I_mask_values).to(
                self.device
            )
            self.Id = utils.sparse_diagonal(
                torch.ones(num_nodes * self.d),
                shape=(num_nodes * self.d, num_nodes * self.d),
            ).to(self.device)

        minus_L = minus_L.coalesce()
        # this help us changing the sign of the elements in the block diagonal
        # with an efficient lower=memory mask
        minus_L = torch.sparse_coo_tensor(
            minus_L.indices(), minus_L.values(), minus_L.size()
        )
        minus_L = minus_L - 2 * minus_L.mul(self.I_mask)
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(
            minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x
        )
        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        return out


class HyperDiffusionOrthoSheafConv(MessagePassing):
    """
    One layer of Sheaf Diffusion with orthogonal Laplacian Y = (I-D^-1/2LD^-1) with L
    normalised with B^-1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        d,
        device,
        dropout=0,
        bias=True,
        norm_type="degree_norm",
        left_proj=None,
        norm=None,
        residual=False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type = norm_type
        self.norm = norm

        # for ortho matrix block <=> degree
        if self.norm_type == "block_norm":
            self.norm_type = "degree_norm"
        elif self.norm_type == "sym_block_norm":
            self.norm_type = "sym_degree_norm"

        self.left_proj = left_proj
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(
                in_channels=d,
                hidden_channels=d,
                out_channels=d,
                num_layers=1,
                dropout=dropout,
                normalisation="ln",
                input_norm=self.norm,
            )
        self.lin = MLP(
            in_channels=in_channels,
            hidden_channels=d,
            out_channels=out_channels,
            num_layers=1,
            dropout=dropout,
            normalisation="ln",
            input_norm=self.norm,
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.device = device

        self.I_mask = None
        self.Id = None
        self.reset_parameters()

    # to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(
        self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1, num_nodes * self.d).t()
        x = self.lin(x)
        data_x = x

        if self.I_mask is None:  # prepare these in advance
            I_mask_indices = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0
            )
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(
                self.device
            )
            self.Id = utils.sparse_diagonal(
                torch.ones(num_nodes * self.d),
                shape=(num_nodes * self.d, num_nodes * self.d),
            ).to(self.device)

        D_inv, B_inv = normalisation_matrices(
            x,
            hyperedge_index,
            alpha,
            num_nodes,
            num_edges,
            self.d,
            norm_type=self.norm_type,
        )

        if self.norm_type in ["sym_degree_norm", "sym_block_norm"]:
            # compute D^(-1/2) @ X
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(
            hyperedge_index, alpha, size=(num_nodes * self.d, num_edges * self.d)
        )
        H_t = torch.sparse.FloatTensor(
            hyperedge_index.flip([0]),
            alpha,
            size=(num_edges * self.d, num_nodes * self.d),
        )

        # these are still diagonal because of ortho
        B_inv = utils.sparse_diagonal(
            B_inv, shape=(num_edges * self.d, num_edges * self.d)
        )
        D_inv = utils.sparse_diagonal(
            D_inv, shape=(num_nodes * self.d, num_nodes * self.d)
        )

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(
            B_inv.indices(),
            B_inv.values(),
            H_t.indices(),
            H_t.values(),
            B_inv.shape[0],
            B_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            H.indices(),
            H.values(),
            minus_L[0],
            minus_L[1],
            H.shape[0],
            H.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            D_inv.indices(),
            D_inv.values(),
            minus_L[0],
            minus_L[1],
            D_inv.shape[0],
            D_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch.sparse_coo_tensor(
            minus_L[0], minus_L[1], size=(num_nodes * self.d, num_nodes * self.d)
        ).to(self.device)

        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(
            minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x
        )

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x
        return out


class HyperDiffusionGeneralSheafConv(MessagePassing):
    """
    One layer of Sheaf Diffusion with general/lowrank Laplacian Y = (I-D^-1/2LD^-1)
    with L normalised with B^-1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        d,
        device,
        dropout=0,
        bias=True,
        norm_type="degree_norm",
        left_proj=None,
        norm=None,
        residual=False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm = norm

        self.left_proj = left_proj
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(
                in_channels=d,
                hidden_channels=d,
                out_channels=d,
                num_layers=1,
                dropout=dropout,
                normalisation="ln",
                input_norm=self.norm,
            )

        self.lin = MLP(
            in_channels=in_channels,
            hidden_channels=d,
            out_channels=out_channels,
            num_layers=1,
            dropout=dropout,
            normalisation="ln",
            input_norm=self.norm,
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.device = device

        self.I_mask = None
        self.Id = None

        self.norm_type = norm_type

        self.reset_parameters()

    # to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    # this is just for block and sym_block normalisation since the matrices D^-1 is a proper inverse that need to be computed
    def normalise(
        self, h_general_sheaf, hyperedge_index, norm_type, num_nodes, num_edges
    ):
        # this is just for block and sym_block normalisation

        # index correspond to the small matrix
        row_small = hyperedge_index[0].view(-1, self.d, self.d)[:, 0, 0] // self.d
        h_general_sheaf_1 = h_general_sheaf.reshape(row_small.shape[0], self.d, self.d)

        to_be_inv_nodes = torch.bmm(
            h_general_sheaf_1, h_general_sheaf_1.permute(0, 2, 1)
        )
        to_be_inv_nodes = scatter_add(
            to_be_inv_nodes, row_small, dim=0, dim_size=num_nodes
        )

        if norm_type in ["block_norm"]:
            d_inv_nodes = utils.batched_sym_matrix_pow(
                to_be_inv_nodes, -1.0
            )  # n_nodes x d x d
            return d_inv_nodes

        elif norm_type in ["sym_block_norm"]:
            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(
                to_be_inv_nodes, -0.5
            )  # n_nodes x d x d
            return d_sqrt_inv_nodes

    def forward(
        self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges
    ) -> Tensor:
        r"""
        Args:
            Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1, num_nodes * self.d).t()

        x = self.lin(x)
        data_x = x

        if self.I_mask is None:  # prepare these in advance
            # I_block = torch.block_diag(*[torch.ones((self.d, self.d)) for i in range(num_nodes)]).to(self.device)
            I_mask_indices = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0
            )
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(
                self.device
            )
            self.Id = utils.sparse_diagonal(
                torch.ones(num_nodes * self.d),
                shape=(num_nodes * self.d, num_nodes * self.d),
            ).to(self.device)

        if self.norm_type in ["block_norm", "sym_block_norm"]:
            # NOTE: the normalisation is specific to general sheaf
            # D_e is the same as before
            B_inv_flat = scatter_add(
                x.new_ones(hyperedge_index.size(1)),
                hyperedge_index[1],
                dim=0,
                dim_size=num_edges * self.d,
            )
            B_inv_flat = 1.0 / B_inv_flat
            B_inv_flat[B_inv_flat == float("inf")] = 0
            B_inv = utils.sparse_diagonal(
                B_inv_flat, shape=(num_edges * self.d, num_edges * self.d)
            )

            # D_v is a dxd matrix than needs to be inverted
            D_inv = self.normalise(
                alpha, hyperedge_index, self.norm_type, num_nodes, num_edges
            )  # num_nodes x d x d
            diag_indices_D = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0
            )
            D_inv_indices = utils.generate_indices_general(diag_indices_D, self.d).to(
                x.device
            )
            D_inv_flat = D_inv.reshape(-1)
            D_inv = torch.sparse.FloatTensor(D_inv_indices, D_inv_flat)

        else:
            D_inv, B_inv = normalisation_matrices(
                x,
                hyperedge_index,
                alpha,
                num_nodes,
                num_edges,
                self.d,
                norm_type=self.norm_type,
            )

        # compute D^(-1/2) @ X for the sym case
        # x: (num_nodes*d) x f
        if self.norm_type == "sym_degree_norm":
            x = D_inv.unsqueeze(-1) * x
        elif self.norm_type == "sym_block_norm":
            D_inv = D_inv.coalesce()
            x = torch_sparse.spmm(
                D_inv.indices(), D_inv.values(), D_inv.shape[0], D_inv.shape[1], x
            )

        if self.norm_type in ["sym_degree_norm", "degree_norm"]:
            # these are still diagonal because of ortho
            B_inv = utils.sparse_diagonal(
                B_inv, shape=(num_edges * self.d, num_edges * self.d)
            )
            D_inv = utils.sparse_diagonal(
                D_inv, shape=(num_nodes * self.d, num_nodes * self.d)
            )

        H = torch.sparse.FloatTensor(
            hyperedge_index, alpha, size=(num_nodes * self.d, num_edges * self.d)
        )
        H_t = torch.sparse.FloatTensor(
            hyperedge_index.flip([0]),
            alpha,
            size=(num_edges * self.d, num_nodes * self.d),
        )

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(
            B_inv.indices(),
            B_inv.values(),
            H_t.indices(),
            H_t.values(),
            B_inv.shape[0],
            B_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            H.indices(),
            H.values(),
            minus_L[0],
            minus_L[1],
            H.shape[0],
            H.shape[1],
            H_t.shape[1],
        )
        minus_L = torch_sparse.spspmm(
            D_inv.indices(),
            D_inv.values(),
            minus_L[0],
            minus_L[1],
            D_inv.shape[0],
            D_inv.shape[1],
            H_t.shape[1],
        )
        minus_L = torch.sparse_coo_tensor(
            minus_L[0], minus_L[1], size=(num_nodes * self.d, num_nodes * self.d)
        ).to(self.device)
        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(
            minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x
        )

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        F = self.out_channels
        out = norm_i.view(-1, 1) * x_j.view(-1, F)
        if alpha is not None:
            out = alpha.view(-1, 1) * out

        return out
