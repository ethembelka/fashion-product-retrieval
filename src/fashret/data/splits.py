"""pair_id-level splits and gallery / query / open-set list construction.

Rules live in docs/EVALUATION_PROTOCOL.md:
  gallery  : source == shop, style > 0
  queries  : source == user, style > 0, at least one gallery item with the same match_key
  openset  : source == user, style == 0
Test split uses the official val_query.json / val_gallery.json instead of these rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ListRule
from .df2_parser import match_key

LIST_COLUMNS = [
    "item_id", "image_id", "pair_id", "style", "match_key", "category_id",
    "scale", "occlusion", "zoom_in", "viewpoint", "crop_path",
]


def split_pair_ids(pair_ids: np.ndarray, val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    """Shuffle unique pair_ids with a fixed seed; last `val_ratio` share -> val_internal."""
    uniq = np.array(sorted(set(int(p) for p in pair_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_ratio))
    val = set(uniq[:n_val].tolist())
    fit = set(uniq[n_val:].tolist())
    assert not (fit & val), "pair_id leak between train_fit and val_internal"
    return fit, val


def apply_rule(items: pd.DataFrame, rule: ListRule) -> pd.DataFrame:
    m = items["source"] == rule.source
    if rule.style_min is not None:
        m &= items["style"] >= rule.style_min
    if rule.style_eq is not None:
        m &= items["style"] == rule.style_eq
    return items[m]


def build_lists(items: pd.DataFrame, gallery: ListRule, queries: ListRule, openset: ListRule) -> dict[str, pd.DataFrame]:
    """Build gallery / queries / openset tables from one split's items (already cropped)."""
    items = items[items["cropped"]]
    g = apply_rule(items, gallery)
    q = apply_rule(items, queries)
    if queries.require_match:
        q = q[q["match_key"].isin(set(g["match_key"]))]
    o = apply_rule(items, openset)
    return {k: v[LIST_COLUMNS].reset_index(drop=True) for k, v in {"gallery": g, "queries": q, "openset": o}.items()}


def official_test_lists(val_query: list[dict], val_gallery: list[dict], items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Map the official DF2 retrieval lists onto our validation items (by image_id + bbox).

    Official records: query -> query_image_id, pair_id, style, bbox, cls ; gallery -> gallery_image_id, ...
    Items whose box is not found (or was not cropped) are dropped and counted by the caller.
    """
    idx = {
        (r.image_id, r.x1, r.y1, r.x2, r.y2): r
        for r in items[items["split"] == "validation"].itertuples(index=False)
    }

    def lookup(recs: list[dict], id_field: str) -> pd.DataFrame:
        rows, missing = [], 0
        for r in recs:
            image_id = f"{int(r[id_field]):06d}"
            key = (image_id, *[int(v) for v in r["bbox"]])
            it = idx.get(key)
            if it is None or not it.cropped:
                missing += 1
                continue
            d = it._asdict()
            d["pair_id"], d["style"] = int(r["pair_id"]), int(r["style"])
            d["match_key"] = match_key(d["pair_id"], d["style"])
            rows.append(d)
        df = pd.DataFrame(rows)
        df.attrs["missing"] = missing
        return df[LIST_COLUMNS]

    return {"queries": lookup(val_query, "query_image_id"), "gallery": lookup(val_gallery, "gallery_image_id")}


def relativize(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    """Store crop_path relative to the processed dir (posix) so lists work on Colab / Drive too."""
    out = df.copy()
    out["crop_path"] = [Path(p).resolve().relative_to(base.resolve()).as_posix() for p in out["crop_path"]]
    return out


def write_lists(lists: dict[str, pd.DataFrame], out_dir: Path, base: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in lists.items():
        if base is not None:
            df = relativize(df, base)
        df.to_csv(out_dir / f"{name}.csv", index=False)
