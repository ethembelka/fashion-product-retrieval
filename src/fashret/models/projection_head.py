"""MLP projection head on top of frozen backbone embeddings: D -> hidden -> out, L2-normalised output."""

from __future__ import annotations

import torch
from torch import nn

from .base import l2_normalize


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.net(x))
