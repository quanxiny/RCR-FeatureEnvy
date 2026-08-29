from __future__ import annotations

from torch import nn


class PairProjectionHead(nn.Sequential):
    def __init__(self, pair_dim: int = 128, hidden_dim: int = 128,
                 projection_dim: int = 64):
        super().__init__(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
