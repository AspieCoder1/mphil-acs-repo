#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import Data

from ..sheaf_hgnn.models import SheafHyperGNN

EPS = 1e-15
MAX_LOGSTD = 10


class VSHAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sheaf_type: str, **kwargs):
        super(VSHAE, self).__init__()
        self.encoder = SheafHyperGNN(
            in_channels=in_channels,
            out_channels=out_channels,
            sheaf_type=sheaf_type,
            is_vshae=True,
            **kwargs
        )
        self.mu_encoder = nn.Linear(self.encoder.out_dim, 128)
        self.logstd_encoder = nn.Linear(self.encoder.out_dim, 128)
        self.mu = None
        self.logstd = None

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.mu_encoder.reset_parameters()
        self.logstd_encoder.reset_parameters()

    def reparametrise(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, data: Data):
        self.mu, self.logstd = self.encoder(data)
        self.logstd = self.logstd.clamp(max=MAX_LOGSTD)
        return self.reparametrise(self.mu, self.logstd)

    def loss(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets)
        kl_divergence = -0.5 * torch.mean(
            torch.sum(
                torch.sum(
                    1 + 2 * self.logstd - self.mu**2 - self.logstd.exp() ** 2, dim=1
                )
            )
        )

        return BCE_loss + kl_divergence

    def __repr__(self):
        return "VSHAE"
