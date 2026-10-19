"""Recall@1 broken down by the query garment's condition labels and category."""

from __future__ import annotations

import pandas as pd

# DeepFashion2 label meanings (dataset README).
LABELS = {
    "scale": {1: "small", 2: "moderate", 3: "large"},
    "occlusion": {1: "slight", 2: "medium", 3: "heavy"},
    "zoom_in": {1: "no", 2: "medium", 3: "large"},
    "viewpoint": {1: "no wear", 2: "frontal", 3: "side/back"},
    "category_id": {
        1: "short sleeve top", 2: "long sleeve top", 3: "short sleeve outwear", 4: "long sleeve outwear",
        5: "vest", 6: "sling", 7: "shorts", 8: "trousers", 9: "skirt", 10: "short sleeve dress",
        11: "long sleeve dress", 12: "vest dress", 13: "sling dress",
    },
}
ATTRIBUTES = ["occlusion", "scale", "zoom_in", "viewpoint", "category_id"]


def breakdown_table(per_query: pd.DataFrame, min_n: int = 30, ks: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    """One row per (attribute, value): n, recall@k for k in ks, small-group flag."""
    rows = []
    for attr in ATTRIBUTES:
        for val, g in per_query.groupby(attr):
            row = {"attribute": attr, "value": int(val), "label": LABELS[attr].get(int(val), str(val)), "n": int(len(g))}
            for k in ks:
                row[f"recall@{k}"] = float(((g.first_hit_rank >= 1) & (g.first_hit_rank <= k)).mean())
            row["small_group"] = len(g) < min_n
            rows.append(row)
    return pd.DataFrame(rows)


def breakdown_dict(table: pd.DataFrame) -> dict:
    """metrics.json layout: breakdown.<attribute>.<label>.{n, recall@1}."""
    out = {}
    for _, r in table.iterrows():
        out.setdefault(r["attribute"], {})[r["label"]] = {"n": int(r["n"]), "recall@1": round(float(r["recall@1"]), 4)}
    return out
