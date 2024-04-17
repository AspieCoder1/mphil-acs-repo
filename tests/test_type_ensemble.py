#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import torch
from models.sheaf_gnn.sheaf_models import TypeEnsembleSheafLearner


def test_type_ensemble():
    x = torch.rand(10, 5)
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 1, 3, 4, 1, 2, 4], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]]
    ).to(torch.int64)
    edge_type = torch.randint(low=0, high=4, size=(edge_index.shape[1],))
    node_type = torch.randint(low=0, high=4, size=(x.shape[0],))

    module = TypeEnsembleSheafLearner(5, (5,), "id", 4, 4)
    module.eval()

    with torch.no_grad():
        out = module(x, edge_index, edge_type, node_type)

    src, dst = edge_index
    x_src = torch.index_select(x, dim=0, index=src)
    x_dst = torch.index_select(x, dim=0, index=dst)

    x_cat = torch.cat(
        [x_src, x_dst],
        dim=1,
    )

    results = []
    for i, x_ij in enumerate(x_cat):
        results.append(module.linear1[edge_type[i].item()](x_ij))

    out1 = torch.row_stack(results)

    assert torch.allclose(out, out1, atol=1e-6)