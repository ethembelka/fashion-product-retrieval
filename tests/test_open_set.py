"""Hand-computed examples for the open-set helpers."""

import numpy as np
import pandas as pd

from fashret.evaluation.breakdown import breakdown_table
from fashret.evaluation.open_set import (
    apply_tau, auroc_known_vs_unknown, coverage_precision_curve, precision_at_coverage, rejection_scores, tau_for_coverage,
)


def test_rejection_scores():
    s = rejection_scores(np.array([0.9, 0.5]), np.array([0.8, 0.1]))
    assert s["abs"].tolist() == [0.9, 0.5]
    assert np.allclose(s["margin"], [0.1, 0.4])


def test_auroc_by_hand():
    # perfectly separated -> 1.0; fully overlapping identical -> 0.5
    assert auroc_known_vs_unknown(np.array([0.9, 0.8]), np.array([0.2, 0.1])) == 1.0
    assert auroc_known_vs_unknown(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
    # 2 known {0.7, 0.3}, 2 unknown {0.5, 0.1}: pairs (k>u): (0.7,0.5),(0.7,0.1),(0.3,0.1) = 3 of 4 -> 0.75
    assert auroc_known_vs_unknown(np.array([0.7, 0.3]), np.array([0.5, 0.1])) == 0.75


def test_tau_and_apply():
    known = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    correct = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1], bool)
    unknown = np.array([0.15, 0.35, 0.55, 0.75])
    tau = tau_for_coverage(known, 0.8)          # accept 80% -> τ at the 20th percentile = 0.28
    r = apply_tau(known, correct, unknown, tau)
    assert r["coverage"] == 0.8                  # 0.3..1.0 accepted (8 of 10)
    assert r["precision"] == 6 / 8               # correct among accepted: 0.4,0.6,0.7,0.8,0.9,1.0
    assert r["false_match_rate"] == 0.75         # unknown >= τ: 0.35, 0.55, 0.75
    curve = coverage_precision_curve(known, correct, unknown, n_points=50)
    assert curve.coverage.is_monotonic_decreasing
    assert abs(precision_at_coverage(curve, 0.8) - 6 / 8) < 1e-9


def test_breakdown_table_flags_small_groups():
    df = pd.DataFrame({
        "occlusion": [1, 1, 2, 2], "scale": [1] * 4, "zoom_in": [1] * 4, "viewpoint": [2] * 4, "category_id": [1, 1, 9, 9],
        "first_hit_rank": [1, 0, 3, 1],
    })
    t = breakdown_table(df, min_n=3)
    occ = t[t.attribute == "occlusion"].set_index("label")
    assert occ.loc["slight", "recall@1"] == 0.5 and occ.loc["medium", "recall@1"] == 0.5
    assert occ.loc["medium", "recall@5"] == 1.0
    assert bool(occ.loc["slight", "small_group"]) is True
