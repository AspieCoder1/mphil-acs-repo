#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import pytest
import torch

from dti_prediction import utils, NODE_TYPE_NAMES, EDGE_TYPE_NAMES, EDGE_TYPE_MAP


class TestGenerateHyperedgeIndex:

    @pytest.fixture
    def incidence_matrices(self):
        return [
            ("drug_disease", torch.Tensor([[1, 1], [0, 1]])),
            ("drug_protein", torch.Tensor([[1, 1], [1, 1]])),
            ("protein_disease", torch.Tensor([[0, 1], [1, 1]])),
        ]

    def test_generates_correct_hyperedge_index(self, incidence_matrices):
        result = utils.generate_hyperedge_index(
            incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
        )

        expected_idx = torch.Tensor(
            [
                [0, 0, 1, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 4, 5, 5, 2, 3, 3],
                [0, 1, 1, 2, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 9, 8, 9, 11, 10, 11],
            ]
        ).to(torch.long)

        assert torch.allclose(result.hyperedge_index, expected_idx)

    def test_correct_hyperedge_types(self, incidence_matrices):
        result = utils.generate_hyperedge_index(
            incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
        )

        expected_hyperedge_types = torch.Tensor([0, 0, 4, 4, 1, 1, 2, 2, 3, 3, 5, 5])

        assert torch.allclose(result.hyperedge_types, expected_hyperedge_types)

    def test_correct_node_types(self, incidence_matrices):
        result = utils.generate_hyperedge_index(
            incidence_matrices, EDGE_TYPE_MAP, EDGE_TYPE_NAMES, NODE_TYPE_NAMES
        )

        expected_node_types = torch.Tensor([0, 0, 2, 2, 1, 1])

        assert torch.allclose(result.node_types, expected_node_types)
