#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_mean

from .config import SheafHGNNConfig

#  This part is for HyperGCN
from .hgcn_sheaf_laplacians import *
from .layers import *
from .sheaf_builder import (
    SheafBuilderDiag,
    SheafBuilderOrtho,
    SheafBuilderGeneral,
    HGCNSheafBuilderDiag,
    HGCNSheafBuilderGeneral,
    HGCNSheafBuilderOrtho,
    HGCNSheafBuilderLowRank,
    SheafBuilderLowRank,
)
from ..hgnn_baselines.mlp import MLP


class SheafHyperGNN(nn.Module):
    """
    This is a Hypergraph Sheaf Model with
    the dxd blocks in H_BIG associated to each pair (node, hyperedge)
    being **diagonal**

    """

    def __init__(self, args: SheafHGNNConfig, sheaf_type: str):
        super(SheafHyperGNN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden
        self.d = args.heads  # dimension of the stalks
        self.init_hedge = (
            args.init_hedge
        )  # how to initialise hyperedge attributes: avg or rand
        self.norm_type = (
            args.sheaf_normtype
        )  # type of laplacian normalisation degree_norm or block_norm
        self.act = (
            args.sheaf_act
        )  # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj  # multiply with (I x W_1) to the left
        self.args = args
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = (
            args.dynamic_sheaf
        )  # if True, theb sheaf changes from one layer to another
        self.residual = args.residual_HCHA

        self.hyperedge_attr = None
        if args.cuda in [0, 1]:
            self.device = torch.device(
                "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        self.lin = MLP(
            in_channels=self.num_features,
            hidden_channels=args.MLP_hidden,
            out_channels=self.MLP_hidden * self.d,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=False,
        )
        self.use_lin2 = args.use_lin2

        # define the model and sheaf generator according to the type of sheaf wanted
        # The diuffusion does not change, however tha implementation for diag and ortho is more efficient
        if sheaf_type == "DiagSheafs":
            ModelSheaf, ModelConv = SheafBuilderDiag, HyperDiffusionDiagSheafConv
        elif sheaf_type == "OrthoSheafs":
            ModelSheaf, ModelConv = SheafBuilderOrtho, HyperDiffusionOrthoSheafConv
        elif sheaf_type == "GeneralSheafs":
            ModelSheaf, ModelConv = SheafBuilderGeneral, HyperDiffusionGeneralSheafConv
        elif sheaf_type == "LowRankSheafs":
            ModelSheaf, ModelConv = SheafBuilderLowRank, HyperDiffusionGeneralSheafConv

        self.convs = nn.ModuleList()
        # Sheaf Diffusion layers
        self.convs.append(
            ModelConv(
                self.MLP_hidden,
                self.MLP_hidden,
                d=self.d,
                device=self.device,
                norm_type=self.norm_type,
                left_proj=self.left_proj,
                norm=self.norm,
                residual=self.residual,
            )
        )

        # Model to generate the reduction maps
        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(ModelSheaf(args))

        for _ in range(self.num_layers - 1):
            # Sheaf Diffusion layers
            self.convs.append(
                ModelConv(
                    self.MLP_hidden,
                    self.MLP_hidden,
                    d=self.d,
                    device=self.device,
                    norm_type=self.norm_type,
                    left_proj=self.left_proj,
                    norm=self.norm,
                    residual=self.residual,
                )
            )
            # Model to generate the reduction maps if the sheaf changes from one layer to another
            if self.dynamic_sheaf:
                self.sheaf_builder.append(ModelSheaf(args))

        self.out_dim = self.MLP_hidden * self.d
        self.lin2 = Linear(self.MLP_hidden * self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

        self.lin.reset_parameters()
        self.lin2.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        # initialize hyperedge attributes either random or as the average of the nodes
        if type == "rand":
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == "avg":
            hyperedge_attr = scatter_mean(
                x[hyperedge_index[0]], hyperedge_index[1], dim=0
            )
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]  # data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        # if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            if data.hyperedge_attr:
                self.hyperedge_attr = data.hyperedge_attr
            else:
                self.hyperedge_attr = self.init_hyperedge_attr(
                    self.init_hedge,
                    num_edges=num_edges,
                    x=x,
                    hyperedge_index=edge_index,
                )

        # expand the input N x num_features -> Nd x num_features such that we can apply the propagation
        x = self.lin(x)
        x = x.view((x.shape[0] * self.d, self.MLP_hidden))  # (N * d) x num_features

        hyperedge_attr = self.lin(self.hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(
            (hyperedge_attr.shape[0] * self.d, self.MLP_hidden)
        )

        for i, conv in enumerate(self.convs[:-1]):
            # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
            if i == 0 or self.dynamic_sheaf:
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i](
                    x, hyperedge_attr, edge_index
                )
            # Sheaf Laplacian Diffusion
            x = F.elu(
                conv(
                    x,
                    hyperedge_index=h_sheaf_index,
                    alpha=h_sheaf_attributes,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
            x = F.dropout(x, p=self.dropout, training=self.training)

        # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
        if len(self.convs) == 1 or self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[-1](
                x, hyperedge_attr, edge_index
            )
        # Sheaf Laplacian Diffusion
        x = self.convs[-1](
            x,
            hyperedge_index=h_sheaf_index,
            alpha=h_sheaf_attributes,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )
        x = x.view(num_nodes, -1)  # Nd x out_channels -> Nx(d*out_channels)
        if self.use_lin2:
            x = self.lin2(x)  # Nx(d*out_channels)-> N x num_classes
        return x


class SheafHyperGCN(nn.Module):
    # replace hyperedge with edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
    def __init__(
        self,
        V,
        num_features,
        num_layers,
        num_classses,
        args: SheafHGNNConfig,
        sheaf_type: str,
    ):
        super(SheafHyperGCN, self).__init__()
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        self.num_nodes = V
        h = [args.MLP_hidden]
        for i in range(l - 1):
            power = l - i + 2
            if (getattr(args, "dname", None) is not None) and args.dname == "citeseer":
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        reapproximate = False  # for HyperGCN we take care of this via dynamic_sheaf

        self.MLP_hidden = args.MLP_hidden
        self.d = args.heads

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden
        self.d = args.heads  # dimension of the stalks
        self.init_hedge = (
            args.init_hedge
        )  # how to initialise hyperedge attributes: avg or rand
        self.norm_type = (
            args.sheaf_normtype
        )  # type of laplacian normalisation degree_norm or block_norm
        self.act = (
            args.sheaf_act
        )  # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj  # multiply with (I x W_1) to the left
        self.args = args
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = (
            args.dynamic_sheaf
        )  # if True, theb sheaf changes from one layer to another
        self.sheaf_type = (
            sheaf_type  #'DiagSheafs', 'OrthoSheafs', 'GeneralSheafs' or 'LowRankSheafs'
        )

        self.hyperedge_attr = None
        self.residual = args.residual_HCHA

        # sheaf_type = 'OrthoSheafs'
        if sheaf_type == "DiagSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderDiag, SheafLaplacianDiag
        elif sheaf_type == "OrthoSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderOrtho, SheafLaplacianOrtho
        elif sheaf_type == "GeneralSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderGeneral, SheafLaplacianGeneral
        elif sheaf_type == "LowRankSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderLowRank, SheafLaplacianGeneral

        if self.left_proj:
            self.lin_left_proj = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.d,
                        hidden_channels=self.d,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        normalisation="ln",
                        input_norm=self.norm,
                    )
                    for i in range(l)
                ]
            )

        self.lin = MLP(
            in_channels=self.num_features,
            hidden_channels=self.MLP_hidden,
            out_channels=self.MLP_hidden * self.d,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=False,
        )

        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(ModelSheaf(args, args.MLP_hidden))

        self.out_dim = h[-1] * self.d
        self.lin2 = Linear(self.out_dim, args.num_classes, bias=False)

        if self.dynamic_sheaf:
            for i in range(1, l):
                self.sheaf_builder.append(ModelSheaf(args, h[i]))

        self.layers = nn.ModuleList(
            [
                utils.HyperGraphConvolution(h[i], h[i + 1], reapproximate, cuda)
                for i in range(l)
            ]
        )
        self.do, self.l = args.dropout, num_layers
        self.m = args.HyperGCN_mediators

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.left_proj:
            for lin_layer in self.lin_left_proj:
                lin_layer.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        # initialize hyperedge attributes either random or as the average of the nodes
        if type == "rand":
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == "avg":
            hyperedge_attr = scatter_mean(
                x[hyperedge_index[0]], hyperedge_index[1], dim=0
            )
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def normalise(self, A, hyperedge_index, num_nodes, d):
        if self.args.sheaf_normtype == "degree_norm":
            # compute D^-1
            D = scatter_add(
                hyperedge_index.new_ones(hyperedge_index.size(1)),
                hyperedge_index[0],
                dim=0,
                dim_size=num_nodes * d,
            )
            D = torch.pow(D, -1.0)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-1A
            A = torch_sparse.spspmm(
                D.indices(),
                D.values(),
                A.indices(),
                A.values(),
                D.shape[0],
                D.shape[1],
                A.shape[1],
            )
            A = torch.sparse_coo_tensor(
                A[0], A[1], size=(num_nodes * d, num_nodes * d)
            ).to(D.device)
        elif self.args.sheaf_normtype == "sym_degree_norm":
            # compute D^-0.5
            D = scatter_add(
                hyperedge_index.new_ones(hyperedge_index.size(1)),
                hyperedge_index[0],
                dim=0,
                dim_size=num_nodes * d,
            )
            D = torch.pow(D, -0.5)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-0.5AD^-0.5
            A = torch_sparse.spspmm(
                D.indices(),
                D.values(),
                A.indices(),
                A.values(),
                D.shape[0],
                D.shape[1],
                A.shape[1],
                coalesced=True,
            )
            A = torch_sparse.spspmm(
                A[0],
                A[1],
                D.indices(),
                D.values(),
                D.shape[0],
                D.shape[1],
                D.shape[1],
                coalesced=True,
            )
            A = torch.sparse_coo_tensor(
                A[0], A[1], size=(num_nodes * d, num_nodes * d)
            ).to(D.device)

        elif self.args.sheaf_normtype == "block_norm":
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0, 2, 1, 3))  # num_nodes x num_nodes x d x d
            D = torch.diagonal(
                D, dim1=0, dim2=1
            )  # d x d x num_nodes (the block diagonal ones)
            D = torch.permute(D, (2, 0, 1))  # num_nodes x d x d

            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = utils.batched_sym_matrix_pow(D, -1.0)  # num_nodes x d x d
            else:
                D = torch.pow(D, -1.0)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D, 0))
            D = D.to_sparse()

            # compute D^-1A
            A = torch.sparse.mm(D, A)  # this is laplacian delta
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1, 1).to_sparse()

        elif self.args.sheaf_normtype == "sym_block_norm":
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0, 2, 1, 3))  # num_nodes x num_nodes x d x d
            D = torch.diagonal(D, dim1=0, dim2=1)  # d x d x num_nodes
            D = torch.permute(D, (2, 0, 1))  # num_nodes x d x d

            # compute D^-1
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = utils.batched_sym_matrix_pow(D, -0.5)  # num_nodes x d x d
            else:
                D = torch.pow(D, -0.5)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D, 0))
            D = D.to_sparse()

            # compute D^-0.5AD^-0.5
            A = torch.sparse.mm(D, A)
            A = torch.sparse.mm(A, D)
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1, 1).to_sparse()
        return A

    def forward(self, data):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        num_nodes = data.x.shape[0]
        num_edges = data.edge_index[1].max().item() + 1

        edge_index = data.edge_index

        if self.hyperedge_attr is None:
            if data.hyperedge_attr:
                self.hyperedge_attr = data.hyperedge_attr
            else:
                self.hyperedge_attr = self.init_hyperedge_attr(
                    self.init_hedge,
                    num_edges=num_edges,
                    x=H,
                    hyperedge_index=edge_index,
                )

        H = self.lin(H)
        hyperedge_attr = self.lin(self.hyperedge_attr)

        H = H.view((H.shape[0] * self.d, self.MLP_hidden))  # (N * d) x num_features
        hyperedge_attr = hyperedge_attr.view(
            (hyperedge_attr.shape[0] * self.d, self.MLP_hidden)
        )

        for i, hidden in enumerate(self.layers):
            if i == 0 or self.dynamic_sheaf:
                # compute the sheaf
                sheaf = self.sheaf_builder[i](
                    H, hyperedge_attr, edge_index
                )  # N x E x d x d

                # build the laplacian based on edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
                # with nondiagonal terms -F_v<e(x_v)^T F_w<e(x_w)
                # and diagonal terms \sum_e F_v<e(x_v)^T F_v<e(x_v)
                h_sheaf_index, h_sheaf_attributes = self.Laplacian(
                    H, m, self.d, edge_index, sheaf
                )

                A = torch.sparse.FloatTensor(
                    h_sheaf_index,
                    h_sheaf_attributes,
                    (num_nodes * self.d, num_nodes * self.d),
                )
                A = A.coalesce()
                A = self.normalise(A, h_sheaf_index, num_nodes, self.d)

                eye_diag = torch.ones((num_nodes * self.d))
                A = (
                    utils.sparse_diagonal(
                        eye_diag, (num_nodes * self.d, num_nodes * self.d)
                    ).to(A.device)
                    - A
                )  # I - A

            if self.left_proj:
                H = H.t().reshape(-1, self.d)
                H = self.lin_left_proj[i](H)
                H = H.reshape(-1, num_nodes * self.d).t()

            H = F.relu(hidden(A, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)

        H = H.view(self.num_nodes, -1)  # Nd x out_channels -> Nx(d*out_channels)
        if self.use_lin2:
            H = self.lin2(H)  # Nx(d*out_channels)-> N x num_classes
        return H
