"""Collect outputs/results/*/metrics.json into comparison tables (CSV + LaTeX)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = ["recall@1", "recall@5", "recall@10", "recall@20", "map", "category_consistency@5"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        eff = m.get("efficiency", {})
        rows.append({
            "run_id": p.parent.name, "model": m.get("model"), "variant": m.get("variant"), "split": m.get("split"),
            **{c: m.get(c) for c in COLUMNS},
            "embed_dim": eff.get("embed_dim"), "search_ms": eff.get("search_ms_per_query"), "index_mb": eff.get("index_mb"),
            "n_queries": m.get("n_queries"), "n_gallery": m.get("n_gallery"),
            "auroc_abs": m.get("openset", {}).get("auroc_abs"), "auroc_margin": m.get("openset", {}).get("auroc_margin"),
        })
    return pd.DataFrame(rows)


def latest_per_key(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest run for each (model, variant, split)."""
    return df.sort_values("run_id").groupby(["model", "variant", "split"], as_index=False).last()


def backbone_table(df: pd.DataFrame, split: str = "test", variant: str = "zeroshot") -> pd.DataFrame:
    t = df[(df.split == split) & (df.variant == variant)].copy()
    t = t.sort_values("recall@1", ascending=False)
    return t[["model", "embed_dim"] + COLUMNS + ["search_ms", "index_mb"]].reset_index(drop=True)


def to_latex(t: pd.DataFrame, caption: str, label: str) -> str:
    cols = [c for c in t.columns]
    fmt = {c: (lambda v: f"{v:.3f}") for c in COLUMNS}
    body = t.to_latex(index=False, float_format="%.3f", formatters=fmt, escape=True, column_format="l" + "r" * (len(cols) - 1))
    return f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n"
