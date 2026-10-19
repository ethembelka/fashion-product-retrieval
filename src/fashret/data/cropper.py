"""Crop each garment from its ground-truth box, with margin, and write a resized JPEG.

Crops are written to `<crops_dir>/<split>/<item_id>.jpg`. Existing crops are skipped,
so the job can be restarted. Boxes smaller than `min_side_px` are not cropped and are
reported in the returned DataFrame (`cropped == False`).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import CropCfg


def expand_box(x1: int, y1: int, x2: int, y2: int, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Grow the box by `margin` of its own size on every side, clipped to the image."""
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(round(bw * margin)), int(round(bh * margin))
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short:
        return img
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def _crop_one(args: tuple[dict, str, CropCfg]) -> tuple[str, bool, str]:
    """Worker: returns (item_id, cropped, reason)."""
    row, out_path, cfg = args
    out = Path(out_path)
    if out.exists():
        return row["item_id"], True, "exists"
    if min(row["x2"] - row["x1"], row["y2"] - row["y1"]) < cfg.min_side_px:
        return row["item_id"], False, "too_small"
    try:
        with Image.open(row["image_path"]) as im:
            im = im.convert("RGB")
            box = expand_box(row["x1"], row["y1"], row["x2"], row["y2"], cfg.margin, *im.size)
            crop = resize_short_side(im.crop(box), cfg.resize_short)
            out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out, "JPEG", quality=cfg.jpeg_quality)
        return row["item_id"], True, "ok"
    except Exception as e:  # corrupt image etc. — report, don't abort the whole run
        return row["item_id"], False, f"error:{type(e).__name__}"


def crop_path(crops_dir: Path, split: str, item_id: str) -> Path:
    return crops_dir / split / f"{item_id}.jpg"


def crop_items(items: pd.DataFrame, crops_dir: Path, cfg: CropCfg, workers: int = 8) -> pd.DataFrame:
    """Crop every row of `items`. Returns items + `crop_path`, `cropped`, `crop_status` columns."""
    paths = [str(crop_path(crops_dir, s, i)) for s, i in zip(items["split"], items["item_id"])]
    jobs = [(row, p, cfg) for row, p in zip(items.to_dict("records"), paths)]
    results: dict[str, tuple[bool, str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for item_id, ok, reason in tqdm(ex.map(_crop_one, jobs, chunksize=64), total=len(jobs), desc="crop"):
            results[item_id] = (ok, reason)
    out = items.copy()
    out["crop_path"] = paths
    out["cropped"] = [results[i][0] for i in out["item_id"]]
    out["crop_status"] = [results[i][1] for i in out["item_id"]]
    return out
