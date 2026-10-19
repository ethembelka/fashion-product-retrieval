"""DeepFashion2 annotation JSON -> one row per garment item.

Each `<split>/annos/<image_id>.json` holds image-level `source` / `pair_id` and
`item1`, `item2`, ... dicts with box, category, style and the four condition labels.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ITEM_COLUMNS = [
    "item_id", "split", "image_id", "image_path", "source", "pair_id", "style",
    "category_id", "category_name", "x1", "y1", "x2", "y2",
    "scale", "occlusion", "zoom_in", "viewpoint",
]


def match_key(pair_id: int, style: int) -> str:
    """Identity of a product: same pair_id AND same style."""
    return f"{pair_id}_{style}"


def parse_anno_file(args: tuple[Path, str, Path]) -> list[dict]:
    """Parse one JSON. Returns a list of item rows (one per garment)."""
    anno_path, split, image_dir = args
    with open(anno_path, encoding="utf-8") as f:
        a = json.load(f)
    image_id = anno_path.stem
    rows = []
    for key, item in a.items():
        if not key.startswith("item"):
            continue
        x1, y1, x2, y2 = item["bounding_box"]
        rows.append({
            "item_id": f"{split}_{image_id}_{key}",
            "split": split,
            "image_id": image_id,
            "image_path": str(image_dir / f"{image_id}.jpg"),
            "source": a["source"],
            "pair_id": int(a["pair_id"]),
            "style": int(item["style"]),
            "category_id": int(item["category_id"]),
            "category_name": item["category_name"],
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "scale": int(item["scale"]),
            "occlusion": int(item["occlusion"]),
            "zoom_in": int(item["zoom_in"]),
            "viewpoint": int(item["viewpoint"]),
        })
    return rows


def parse_split(root: Path, split: str, workers: int = 8) -> pd.DataFrame:
    """Parse every annotation JSON of `<root>/<split>/annos` into an items table."""
    anno_dir = root / split / "annos"
    image_dir = root / split / "image"
    files = sorted(anno_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation JSON under {anno_dir}")
    jobs = [(p, split, image_dir) for p in files]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(parse_anno_file, jobs, chunksize=256), total=len(jobs), desc=f"parse {split}"):
            rows.extend(r)
    df = pd.DataFrame(rows, columns=ITEM_COLUMNS)
    df["match_key"] = [match_key(p, s) for p, s in zip(df["pair_id"], df["style"])]
    return df
