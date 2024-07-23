#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import pytest
import torch

from dti_prediction import utils, NODE_TYPE_NAMES, EDGE_TYPE_NAMES, EDGE_TYPE_MAP


@pytest.fixture
def incidence_matrices():
    return [
        ("drug_disease", torch.Tensor([[1, 1], [0, 1]])),
        ("drug_protein", torch.Tensor([[1, 1], [1, 1]])),
        ("protein_disease", torch.Tensor([[0, 1], [1, 1]])),
    ]


@pytest.fixture
def incidence_matrices_empty_column():
    return [
        ("drug_disease", torch.Tensor([[0, 1], [0, 1]])),
        ("drug_protein", torch.Tensor([[1, 1], [1, 1]])),
        ("protein_disease", torch.Tensor([[0, 1], [1, 1]])),
    ]


@pytest.fixture
def hyperedge_index():
    return torch.Tensor(
        [
            [0, 1, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 4, 5, 5, 2, 3, 3],
            [1, 1, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 9, 8, 9, 11, 10, 11],
        ]
    ).to(torch.long)


def test_generates_correct_hyperedge_index(incidence_matrices):
    result = utils.generate_hyperedge_index(
        incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
    ).hyperedge_index

    expected_idx = torch.Tensor(
        [
            [0, 0, 1, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 4, 5, 5, 2, 3, 3],
            [0, 1, 1, 2, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 9, 8, 9, 11, 10, 11],
        ]
    ).to(torch.long)

    assert torch.allclose(result, expected_idx)


def test_correct_hyperedge_types(incidence_matrices):
    result = utils.generate_hyperedge_index(
        incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
    )

    expected_hyperedge_types = torch.Tensor([0, 0, 4, 4, 1, 1, 2, 2, 3, 3, 5, 5])

    assert torch.allclose(result.hyperedge_types, expected_hyperedge_types)


def test_correct_node_types(incidence_matrices):
    result = utils.generate_hyperedge_index(
        incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
    )

    expected_node_types = torch.Tensor([0, 0, 2, 2, 1, 1])

    assert torch.allclose(result.node_types, expected_node_types)


def test_correct_hyperedge_index_with_empty_column(incidence_matrices_empty_column):
    result = utils.generate_hyperedge_index(
        incidence_matrices_empty_column,
        EDGE_TYPE_MAP,
        EDGE_TYPE_NAMES,
        NODE_TYPE_NAMES,
    ).hyperedge_index

    expected_idx = torch.Tensor(
        [
            [0, 1, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 4, 5, 5, 2, 3, 3],
            [1, 1, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 9, 8, 9, 11, 10, 11],
        ]
    ).to(torch.long)

    assert torch.allclose(expected_idx, result, rtol=1e-6)


def test_generate_incidence_graph():
    hyperedge_index = torch.Tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    incidence_graph = utils.generate_incidence_graph(hyperedge_index)

    expected_incidence_graph = torch.Tensor([[0, 1, 2, 3], [4, 4, 5, 5]]).to(torch.long)

    assert torch.allclose(incidence_graph, expected_incidence_graph)


def test_generate_incidence_graph_realistic(hyperedge_index):
    incidence_graph = utils.generate_incidence_graph(hyperedge_index)
    expected_incidence_graph = hyperedge_index + torch.Tensor([[0], [6]]).to(torch.long)

    assert torch.allclose(incidence_graph, expected_incidence_graph)


def test_generate_node_features(hyperedge_index):
    incidence_graph = utils.generate_incidence_graph(hyperedge_index)
    node_features = utils.generate_node_features(incidence_graph)

    assert node_features.shape == (18, 128)
