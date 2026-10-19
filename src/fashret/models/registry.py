"""Name -> FeatureExtractor, driven by configs/models/backbones.yaml.

Adding a backbone = one YAML entry (+ a new class only if the library is new).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import load_yaml
from .base import FeatureExtractor


def load_backbone_cfg(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_extractor(name: str, cfg: dict[str, Any], device: torch.device | None = None) -> FeatureExtractor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = cfg["backbones"][name]
    fp16 = cfg.get("precision", "fp32") == "fp16"
    lib = spec["library"]
    if lib == "timm":
        from .timm_backbone import TimmExtractor

        return TimmExtractor(name, spec["name"], spec["image_size"], device, fp16)
    if lib == "open_clip":
        from .openclip_backbone import OpenClipExtractor

        return OpenClipExtractor(name, spec["name"], spec["pretrained"], device, fp16)
    raise ValueError(f"Unknown library {lib!r} for backbone {name!r}")
