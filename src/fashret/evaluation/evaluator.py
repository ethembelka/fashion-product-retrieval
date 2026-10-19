"""Evaluation of one model/split feature directory: closed set + open set + breakdowns.

Returns a metrics dict (-> metrics.json), a per-query table and the coverage curves.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..index.faiss_index import FlatIPIndex
from ..utils.io import read_table
from .breakdown import breakdown_dict, breakdown_table
from .closed_set import (
    category_consistency, first_hit_rank, hits_matrix, mean_average_precision, recall_at_k, relevant_counts,
)
from .open_set import (
    apply_tau, auroc_known_vs_unknown, coverage_precision_curve, precision_at_coverage, rejection_scores, tau_for_coverage,
)


def load_list(feat_dir: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(feat_dir / f"{name}_emb.npy")
    meta = read_table(feat_dir / f"{name}_meta.parquet")
    assert len(meta) == emb.shape[0], f"{name}: meta/emb row mismatch"
    return emb, meta


def search_scores(index: FlatIPIndex, q_emb: np.ndarray, g_keys: np.ndarray, search_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scores, idx, s1, s2): s2 = best score with a match_key different from the top-1's."""
    scores, idx = index.search(q_emb, search_k)
    ranked_keys = g_keys[idx]
    diff = ranked_keys != ranked_keys[:, :1]
    s2_idx = np.where(diff.any(axis=1), diff.argmax(axis=1), scores.shape[1] - 1)
    return scores, idx, scores[:, 0], scores[np.arange(len(scores)), s2_idx]


def evaluate_split(feat_dir: Path, recall_k: list[int], search_k: int, consistency_k: int = 5,
                   tau: dict[str, float] | None = None, tau_target_coverage: float = 0.95,
                   report_coverages: tuple[float, ...] = (0.5, 0.8, 0.95), min_group: int = 30) -> dict:
    """`tau` = thresholds chosen on another split ({abs: τ, margin: τ}); None → choose here (val_internal)."""
    g_emb, g_meta = load_list(feat_dir, "gallery")
    q_emb, q_meta = load_list(feat_dir, "queries")
    o_emb, o_meta = load_list(feat_dir, "openset")

    index = FlatIPIndex(g_emb.shape[1])
    index.add(g_emb)
    g_keys, g_cats = g_meta["match_key"].to_numpy(), g_meta["category_id"].to_numpy()

    t0 = time.time()
    scores, idx, q_s1, q_s2 = search_scores(index, q_emb, g_keys, search_k)
    search_ms = (time.time() - t0) * 1000 / len(q_emb)
    _, o_idx, o_s1, o_s2 = search_scores(index, o_emb, g_keys, search_k)

    # ---- closed set
    ranked_keys = g_keys[idx]
    hits = hits_matrix(ranked_keys, q_meta["match_key"].to_numpy())
    n_rel = relevant_counts(q_meta["match_key"].to_numpy(), g_keys)
    metrics: dict = recall_at_k(hits, recall_k)
    metrics["map"] = mean_average_precision(hits, n_rel)
    metrics[f"category_consistency@{consistency_k}"] = category_consistency(g_cats[idx], q_meta["category_id"].to_numpy(), consistency_k)
    metrics[f"openset_category_consistency@{consistency_k}"] = category_consistency(g_cats[o_idx], o_meta["category_id"].to_numpy(), consistency_k)
    metrics.update(n_queries=int(len(q_emb)), n_gallery=int(len(g_emb)), n_openset=int(len(o_emb)),
                   queries_without_match=int((n_rel == 0).sum()))

    per_query = q_meta.copy()
    per_query["first_hit_rank"] = first_hit_rank(hits)
    per_query["top1_correct"] = hits[:, 0]
    per_query["top1_key"] = ranked_keys[:, 0]
    per_query["s1"], per_query["s2"] = q_s1, q_s2
    per_open = o_meta.copy()
    per_open["s1"], per_open["s2"] = o_s1, o_s2
    per_open["top1_key"] = g_keys[o_idx[:, 0]]

    # ---- open set
    known, unknown = rejection_scores(q_s1, q_s2), rejection_scores(o_s1, o_s2)
    correct = hits[:, 0]
    openset: dict = {}
    curves: dict[str, pd.DataFrame] = {}
    chosen_tau: dict[str, float] = {}
    for name in ("abs", "margin"):
        openset[f"auroc_{name}"] = auroc_known_vs_unknown(known[name], unknown[name])
        # selective prediction: does the score tell a correct top-1 from a wrong one (known queries only)?
        openset[f"auroc_correct_{name}"] = auroc_known_vs_unknown(known[name][correct], known[name][~correct])
        curve = coverage_precision_curve(known[name], correct, unknown[name])
        curves[name] = curve
        for c in report_coverages:
            openset[f"precision@cov{int(c * 100)}_{name}"] = precision_at_coverage(curve, c)
        t = tau[name] if tau else tau_for_coverage(known[name], tau_target_coverage)
        chosen_tau[name] = t
        openset[f"at_tau_{name}"] = apply_tau(known[name], correct, unknown[name], t)
    openset["tau_source"] = "given" if tau else "this_split"
    openset["tau"] = chosen_tau
    metrics["openset"] = openset

    # ---- breakdowns
    bt = breakdown_table(per_query, min_n=min_group)
    metrics["breakdown"] = breakdown_dict(bt)

    metrics["efficiency"] = {"embed_dim": int(g_emb.shape[1]), "search_ms_per_query": round(search_ms, 4), "index_mb": round(index.size_mb, 2)}
    return {"metrics": metrics, "per_query": per_query, "per_openset": per_open, "curves": curves, "breakdown": bt}
