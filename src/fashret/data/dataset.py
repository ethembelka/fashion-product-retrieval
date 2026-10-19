"""Crop dataset for feature extraction: list CSV -> (tensor, row index)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CropDataset(Dataset):
    def __init__(self, items: pd.DataFrame, processed_dir: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = [processed_dir / p for p in items["crop_path"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), i
