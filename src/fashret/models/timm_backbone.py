"""timm backbones (ResNet-50, DINOv2 ViT-B/14) as FeatureExtractors.

Global descriptor = timm's pooled features (`num_classes=0`): global-average pool for
ResNet, CLS token for DINOv2. Preprocessing follows the model's own pretrained config.
"""

from __future__ import annotations

from typing import Callable

import timm
import torch
from PIL import Image

from .base import l2_normalize


class TimmExtractor:
    def __init__(self, name: str, timm_name: str, image_size: int, device: torch.device, fp16: bool) -> None:
        self.name = name
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        # ViTs (DINOv2 ships at 518 px) must be told the working resolution; CNNs take any size.
        extra = {"img_size": image_size} if "vit" in timm_name else {}
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0, **extra).eval().to(device)
        if self.fp16:
            self.model = self.model.half()
        cfg = timm.data.resolve_data_config({"input_size": (3, image_size, image_size)}, model=self.model)
        # Each backbone keeps its official eval preprocessing (resize + center crop). A uniform
        # "squash to S×S" was tried and hurt ResNet (-3.5 R@1 on a 256-way probe) — see DECISIONS.md.
        self._tf = timm.data.create_transform(**cfg, is_training=False)
        self.embed_dim = int(self.model.num_features)

    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]:
        return self._tf

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(self.device, non_blocking=True)
        if self.fp16:
            x = x.half()
        return l2_normalize(self.model(x).float())
