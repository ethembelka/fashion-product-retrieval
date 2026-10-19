"""Hand-computed examples for Recall@K and mAP."""

import numpy as np

from fashret.evaluation.closed_set import (
    average_precision, category_consistency, first_hit_rank, hits_matrix, mean_average_precision,
    recall_at_k, relevant_counts,
)

# 3 queries, top-4 ranked gallery keys each
RANKED = np.array([
    ["A", "B", "A", "C"],   # q0 (key A): hits at ranks 1 and 3
    ["B", "C", "D", "A"],   # q1 (key A): hit at rank 4
    ["X", "Y", "Z", "W"],   # q2 (key B): no hit
])
QKEYS = np.array(["A", "A", "B"])


def test_hits_and_recall():
    h = hits_matrix(RANKED, QKEYS)
    assert h.tolist() == [[1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    r = recall_at_k(h, [1, 2, 4])
    assert r["recall@1"] == 1 / 3
    assert r["recall@2"] == 1 / 3
    assert r["recall@4"] == 2 / 3


def test_average_precision_by_hand():
    # q0: hits at 1 and 3 with 2 relevant -> (1/1 + 2/3)/2 = 0.8333
    assert abs(average_precision(np.array([1, 0, 1, 0], bool), 2) - (1 + 2 / 3) / 2) < 1e-9
    # same hits but 3 relevant in gallery (one never retrieved) -> (1 + 2/3)/3
    assert abs(average_precision(np.array([1, 0, 1, 0], bool), 3) - (1 + 2 / 3) / 3) < 1e-9
    # q1: single hit at rank 4, 1 relevant -> 0.25
    assert average_precision(np.array([0, 0, 0, 1], bool), 1) == 0.25
    # no hit -> 0
    assert average_precision(np.zeros(4, bool), 1) == 0.0


def test_map_and_rank():
    h = hits_matrix(RANKED, QKEYS)
    n_rel = np.array([2, 1, 1])
    expected = ((1 + 2 / 3) / 2 + 0.25 + 0.0) / 3
    assert abs(mean_average_precision(h, n_rel) - expected) < 1e-9
    assert first_hit_rank(h).tolist() == [1, 4, 0]


def test_relevant_counts_and_category_consistency():
    assert relevant_counts(np.array(["A", "B", "Q"]), np.array(["A", "A", "B"])).tolist() == [2, 1, 0]
    cats = np.array([[1, 1, 2], [3, 1, 1]])
    assert category_consistency(cats, np.array([1, 1]), k=3) == 4 / 6
    assert category_consistency(cats, np.array([1, 1]), k=1) == 0.5
