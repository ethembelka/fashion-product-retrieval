"""Closed-set retrieval metrics on ranked gallery keys.

Inputs are plain arrays so the code is model-agnostic:
  ranked_keys : (n_q, k) match_key of the k best gallery items per query, best first
  query_keys  : (n_q,)   match_key of each query
  n_relevant  : (n_q,)   number of gallery items sharing the query's match_key (for mAP)
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def hits_matrix(ranked_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    """Boolean (n_q, k): is the gallery item at rank j a correct match for query i."""
    return ranked_keys == np.asarray(query_keys)[:, None]


def recall_at_k(hits: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """Share of queries with at least one hit in the first K ranks."""
    return {f"recall@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}


def average_precision(hit_row: np.ndarray, n_relevant: int) -> float:
    """AP over the ranked list; relevant items beyond the list depth count as misses."""
    if n_relevant == 0:
        return 0.0
    ranks = np.flatnonzero(hit_row) + 1  # 1-based ranks of hits
    if ranks.size == 0:
        return 0.0
    precisions = np.arange(1, ranks.size + 1) / ranks
    return float(precisions.sum() / n_relevant)


def mean_average_precision(hits: np.ndarray, n_relevant: np.ndarray) -> float:
    return float(np.mean([average_precision(h, n) for h, n in zip(hits, n_relevant)]))


def first_hit_rank(hits: np.ndarray) -> np.ndarray:
    """1-based rank of the first correct item per query; 0 when none within k."""
    any_hit = hits.any(axis=1)
    r = hits.argmax(axis=1) + 1
    return np.where(any_hit, r, 0)


def relevant_counts(query_keys: np.ndarray, gallery_keys: np.ndarray) -> np.ndarray:
    c = Counter(gallery_keys.tolist())
    return np.array([c.get(k, 0) for k in query_keys], dtype=np.int64)


def category_consistency(ranked_cats: np.ndarray, query_cats: np.ndarray, k: int) -> float:
    """'Similar-alternative quality': share of top-K results in the query's category."""
    return float((ranked_cats[:, :k] == np.asarray(query_cats)[:, None]).mean())
