"""End-to-end data preparation: parse -> crop -> splits -> lists -> summary tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataCfg
from ..utils.io import ensure_dir, read_json, read_table, write_json, write_table
from .cropper import crop_items
from .df2_parser import parse_split
from .splits import build_lists, official_test_lists, split_pair_ids, write_lists


def check_raw_layout(root: Path) -> list[str]:
    needed = [
        "train/image", "train/annos", "validation/image", "validation/annos",
        "validation/json_for_validation/val_query.json", "validation/json_for_validation/val_gallery.json",
    ]
    return [n for n in needed if not (root / n).exists()]


def parse_or_load(root: Path, split: str, processed: Path, workers: int) -> pd.DataFrame:
    path = processed / f"items_{split}.parquet"
    if path.exists():
        return read_table(path)
    df = parse_split(root, split, workers)
    write_table(df, path)
    return df


def crop_or_load(items: pd.DataFrame, split: str, cfg: DataCfg, workers: int) -> pd.DataFrame:
    path = cfg.paths.processed_dir / f"items_{split}_cropped.parquet"
    if path.exists():
        return read_table(path)
    df = crop_items(items, cfg.paths.processed_dir / "crops", cfg.crop, workers)
    write_table(df, path)
    return df


def summarize(lists_by_split: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for split, lists in lists_by_split.items():
        for name, df in lists.items():
            rows.append({
                "split": split, "list": name, "n_items": len(df),
                "n_products": df["match_key"].nunique(), "n_pair_ids": df["pair_id"].nunique(),
            })
    return pd.DataFrame(rows)


def category_distribution(lists_by_split: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for split, lists in lists_by_split.items():
        for name, df in lists.items():
            c = df["category_id"].value_counts().sort_index().rename(f"{split}/{name}")
            frames.append(c)
    return pd.concat(frames, axis=1).fillna(0).astype(int)


def prepare(cfg: DataCfg, workers: int = 8) -> dict:
    root = cfg.paths.deepfashion2_root
    missing = check_raw_layout(root)
    if missing:
        raise FileNotFoundError(f"Missing under {root}: {missing}")
    processed = ensure_dir(cfg.paths.processed_dir)

    items = {s: parse_or_load(root, s, processed, workers) for s in ("train", "validation")}
    cropped = {s: crop_or_load(df, s, cfg, workers) for s, df in items.items()}

    crop_report = {
        s: {
            "total": int(len(df)),
            "cropped": int(df["cropped"].sum()),
            "dropped_too_small": int((df["crop_status"] == "too_small").sum()),
            "errors": int(df["crop_status"].str.startswith("error").sum()),
        }
        for s, df in cropped.items()
    }

    fit_ids, val_ids = split_pair_ids(cropped["train"]["pair_id"].to_numpy(), cfg.splits.internal_val_ratio, cfg.splits.seed)
    tr = cropped["train"]
    by_split = {
        "train_fit": build_lists(tr[tr["pair_id"].isin(fit_ids)], cfg.gallery, cfg.queries, cfg.openset),
        "val_internal": build_lists(tr[tr["pair_id"].isin(val_ids)], cfg.gallery, cfg.queries, cfg.openset),
    }

    jv = root / "validation/json_for_validation"
    test = official_test_lists(read_json(jv / "val_query.json"), read_json(jv / "val_gallery.json"), cropped["validation"])
    crop_report["test_official_missing"] = {k: int(v.attrs.get("missing", 0)) for k, v in test.items()}
    test["openset"] = build_lists(cropped["validation"], cfg.gallery, cfg.queries, cfg.openset)["openset"]
    by_split["test"] = test

    splits_dir = processed / "splits"
    for name, lists in by_split.items():
        write_lists(lists, splits_dir / name, base=processed)
    pd.DataFrame({"pair_id": sorted(fit_ids)}).to_csv(splits_dir / "train_fit_pair_ids.csv", index=False)
    pd.DataFrame({"pair_id": sorted(val_ids)}).to_csv(splits_dir / "val_internal_pair_ids.csv", index=False)

    write_json(crop_report, processed / "crop_report.json")
    summary = summarize(by_split)
    write_table(summary, Path("reports/tables/data_summary.csv"))
    category_distribution(by_split).to_csv(Path("reports/tables/data_categories.csv"))
    return {"crop_report": crop_report, "summary": summary}
