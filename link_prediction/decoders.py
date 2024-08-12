#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredDecoder(nn.Module):
    def __init__(self, dim):
        super(LinkPredDecoder, self).__init__()
        self.dim = dim

    def forward(self, left_emb, right_emb) -> torch.Tensor:
        raise NotImplementedError


class DotProductDecoder(LinkPredDecoder):
    def __init__(self, dim):
        super(DotProductDecoder, self).__init__(dim=dim)

    def forward(self, left_emb, right_emb) -> torch.Tensor:
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()


class DistMultDecoder(LinkPredDecoder):
    def __init__(self, dim: int):
        super(DistMultDecoder, self).__init__(dim=dim)
        self.bilinear = nn.Bilinear(in1_features=dim, in2_features=dim, out_features=1, bias=False)
        nn.init.xavier_normal_(self.bilinear.weight, gain=1.414)

    def forward(self, left_emb, right_emb) -> torch.Tensor:
        return self.bilinear(left_emb, right_emb)


class ConcatDecoder(LinkPredDecoder):
    def __init__(self, dim: int):
        super(ConcatDecoder, self).__init__(dim=dim)
        self.dim = dim
        self.fc = nn.Linear(2 * dim, 1)

    def forward(self, left_emb, right_emb) -> torch.Tensor:
        emb_cat = torch.cat([left_emb, right_emb], dim=1)
        return self.fc(emb_cat)
