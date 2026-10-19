"""Closed-set evaluation of one model/split feature directory -> metrics dict + per-query table.

Open-set scoring and breakdowns are added in later steps; this module stays the single entry point.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..index.faiss_index import FlatIPIndex
from ..utils.io import read_table
from .closed_set import (
    category_consistency, first_hit_rank, hits_matrix, mean_average_precision, recall_at_k, relevant_counts,
)


def load_list(feat_dir: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(feat_dir / f"{name}_emb.npy")
    meta = read_table(feat_dir / f"{name}_meta.parquet")
    assert len(meta) == emb.shape[0], f"{name}: meta/emb row mismatch"
    return emb, meta


def evaluate_closed_set(feat_dir: Path, recall_k: list[int], search_k: int, consistency_k: int = 5) -> tuple[dict, pd.DataFrame, dict]:
    g_emb, g_meta = load_list(feat_dir, "gallery")
    q_emb, q_meta = load_list(feat_dir, "queries")

    index = FlatIPIndex(g_emb.shape[1])
    index.add(g_emb)
    t0 = time.time()
    scores, idx = index.search(q_emb, search_k)
    search_ms = (time.time() - t0) * 1000 / len(q_emb)

    g_keys = g_meta["match_key"].to_numpy()
    g_cats = g_meta["category_id"].to_numpy()
    ranked_keys = g_keys[idx]
    hits = hits_matrix(ranked_keys, q_meta["match_key"].to_numpy())
    n_rel = relevant_counts(q_meta["match_key"].to_numpy(), g_keys)

    metrics = recall_at_k(hits, recall_k)
    metrics["map"] = mean_average_precision(hits, n_rel)
    metrics[f"category_consistency@{consistency_k}"] = category_consistency(g_cats[idx], q_meta["category_id"].to_numpy(), consistency_k)
    metrics["n_queries"] = int(len(q_emb))
    metrics["n_gallery"] = int(len(g_emb))
    metrics["queries_without_match"] = int((n_rel == 0).sum())

    per_query = q_meta.copy()
    per_query["first_hit_rank"] = first_hit_rank(hits)
    per_query["s1"] = scores[:, 0]
    per_query["top1_key"] = ranked_keys[:, 0]
    per_query["top1_correct"] = hits[:, 0]
    # s2 = best score among results with a different match_key than the top-1 (for open-set margin)
    diff = ranked_keys != ranked_keys[:, :1]
    has_diff = diff.any(axis=1)
    s2_idx = np.where(has_diff, diff.argmax(axis=1), scores.shape[1] - 1)
    per_query["s2"] = scores[np.arange(len(scores)), s2_idx]

    efficiency = {"embed_dim": int(g_emb.shape[1]), "search_ms_per_query": round(search_ms, 4), "index_mb": round(index.size_mb, 2)}
    return metrics, per_query, efficiency
