"""FeatureExtractor protocol: every backbone exposes the same two things.

- `preprocess`: PIL image -> tensor (C,H,W) ready for the model.
- `encode`: batch tensor (B,C,H,W) -> L2-normalised embeddings (B,D).
Evaluation code only ever sees the (B,D) arrays, never the model.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch
from PIL import Image


@runtime_checkable
class FeatureExtractor(Protocol):
    name: str
    embed_dim: int

    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]: ...

    def encode(self, batch: torch.Tensor) -> torch.Tensor: ...


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
