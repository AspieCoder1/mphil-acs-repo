import torch

from models.sheaf_gnn.lib.laplace import compute_left_right_map_index, compute_left_right_map_index_old


def test_new_compute_left_right_map_index_full_matrix():
    input = torch.tensor([[0, 1, 5, 4, 2, 3, 6, 3], [2, 3, 6, 3, 0, 1, 5, 4]])
    lr_index, edge_index = compute_left_right_map_index_old(input, True)

    lr_index_new, edge_index_new = compute_left_right_map_index(input, True)

    assert torch.allclose(edge_index, edge_index_new, atol=1e-6)
    assert torch.allclose(lr_index, lr_index_new, atol=1e-6)


def test_new_compute_left_right_map_index_triangular():
    input = torch.tensor([[0, 1, 5, 4, 2, 3, 6, 3], [2, 3, 6, 3, 0, 1, 5, 4]])
    lr_index, edge_index = compute_left_right_map_index_old(input)

    lr_index_new, edge_index_new = compute_left_right_map_index(input)

    assert torch.allclose(edge_index, edge_index_new, atol=1e-6)
    assert torch.allclose(lr_index, lr_index_new, atol=1e-6)
