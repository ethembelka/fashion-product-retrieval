"""YAML config loading into typed dataclasses. No value is hard-coded elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class PathsCfg:
    deepfashion2_root: Path
    processed_dir: Path
    features_dir: Path
    outputs_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path, root: Path | None = None) -> "PathsCfg":
        root = root or Path.cwd()
        d = load_yaml(path)
        return cls(**{k: (root / d[k]).resolve() for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CropCfg:
    margin: float
    min_side_px: int
    resize_short: int
    jpeg_quality: int


@dataclass(frozen=True)
class SplitCfg:
    internal_val_ratio: float
    seed: int


@dataclass(frozen=True)
class ListRule:
    """Selection rule for a gallery / query / open-set list."""

    source: str
    style_min: int | None = None
    style_eq: int | None = None
    require_match: bool = False


@dataclass(frozen=True)
class DataCfg:
    paths: PathsCfg
    crop: CropCfg
    splits: SplitCfg
    gallery: ListRule
    queries: ListRule
    openset: ListRule
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path, root: Path | None = None) -> "DataCfg":
        root = root or Path(path).resolve().parent.parent
        d = load_yaml(path)
        return cls(
            paths=PathsCfg.from_yaml(root / d["paths"], root),
            crop=CropCfg(**d["crop"]),
            splits=SplitCfg(**d["splits"]),
            gallery=ListRule(**d["gallery"]),
            queries=ListRule(**d["queries"]),
            openset=ListRule(**d["openset"]),
            raw=d,
        )
