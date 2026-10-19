"""open_clip backbones (CLIP ViT-B/16, SigLIP ViT-B/16) as FeatureExtractors.

Only the image tower is used: `encode_image` returns the joint-space image embedding.
"""

from __future__ import annotations

from typing import Callable

import open_clip
import torch
from PIL import Image

from .base import l2_normalize


class OpenClipExtractor:
    def __init__(self, name: str, arch: str, pretrained: str, device: torch.device, fp16: bool) -> None:
        self.name = name
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model.eval().to(device)
        if self.fp16:
            self.model = self.model.half()
        self._tf = preprocess
        with torch.inference_mode():
            d = self.model.encode_image(torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float16 if self.fp16 else torch.float32))
        self.embed_dim = int(d.shape[-1])

    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]:
        return self._tf

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(self.device, non_blocking=True)
        if self.fp16:
            x = x.half()
        return l2_normalize(self.model.encode_image(x).float())
