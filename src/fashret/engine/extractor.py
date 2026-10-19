"""Batched feature extraction for one split's lists.

Output per list: `<features_dir>/<model>/<split>/<list>_emb.npy` (float32, L2-normalised,
row order == list CSV order) and `<list>_meta.parquet` (the list rows). Existing files are
skipped unless `overwrite=True`, so a Colab session can resume.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import CropDataset
from ..models.base import FeatureExtractor
from ..utils.io import ensure_dir, write_json, write_table


def extract_list(model: FeatureExtractor, items: pd.DataFrame, processed_dir: Path, batch_size: int,
                 num_workers: int, limit: int | None = None) -> tuple[np.ndarray, dict]:
    if limit:
        items = items.head(limit)
    ds = CropDataset(items, processed_dir, model.preprocess())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    out = np.zeros((len(ds), model.embed_dim), dtype=np.float32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for x, idx in tqdm(dl, desc=f"{model.name}", mininterval=2):
        out[idx.numpy()] = model.encode(x).cpu().numpy()
    dt = time.time() - t0
    stats = {
        "n": len(ds), "seconds": round(dt, 1), "images_per_s": round(len(ds) / max(dt, 1e-6), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0,
    }
    assert np.allclose(np.linalg.norm(out[:8], axis=1), 1.0, atol=1e-3), "embeddings must be L2-normalised"
    return out, stats


def extract_split(model: FeatureExtractor, split: str, lists: list[str], processed_dir: Path, features_dir: Path,
                  batch_size: int, num_workers: int, overwrite: bool = False, limit: int | None = None) -> dict:
    out_dir = ensure_dir(features_dir / model.name / split)
    report: dict = {"model": model.name, "split": split, "embed_dim": model.embed_dim, "lists": {}}
    for name in lists:
        emb_path, meta_path = out_dir / f"{name}_emb.npy", out_dir / f"{name}_meta.parquet"
        if emb_path.exists() and meta_path.exists() and not overwrite:
            report["lists"][name] = {"skipped": True, "n": int(np.load(emb_path, mmap_mode="r").shape[0])}
            continue
        items = pd.read_csv(processed_dir / "splits" / split / f"{name}.csv")
        emb, stats = extract_list(model, items, processed_dir, batch_size, num_workers, limit)
        if limit:
            items = items.head(limit)
        assert len(items) == emb.shape[0]
        np.save(emb_path, emb)
        write_table(items, meta_path)
        report["lists"][name] = stats
    write_json(report, out_dir / "extract_report.json")
    return report
