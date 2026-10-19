"""Open-set ("not in catalog") analysis.

Two rejection scores per query, both "higher = more confident the top-1 is right":
  abs    = s1            (cosine of the best gallery item)
  margin = s1 - s2       (gap to the best gallery item with a DIFFERENT match_key)
Known  = closed-set queries (have a match in the gallery). Unknown = open-set queries (style = 0).

Coverage–precision: sweeping a threshold τ over known-query scores, coverage = share of known
queries accepted (score >= τ), precision = top-1 accuracy among accepted; false-match rate = share
of unknown queries that would be accepted at τ. τ is picked on val_internal for a target coverage
and then applied unchanged on test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rejection_scores(s1: np.ndarray, s2: np.ndarray) -> dict[str, np.ndarray]:
    return {"abs": np.asarray(s1, dtype=np.float64), "margin": np.asarray(s1, dtype=np.float64) - np.asarray(s2, dtype=np.float64)}


def auroc_known_vs_unknown(known: np.ndarray, unknown: np.ndarray) -> float:
    """How well the score separates known (label 1) from unknown (label 0) queries."""
    y = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    return float(roc_auc_score(y, np.r_[known, unknown]))


def coverage_precision_curve(known_score: np.ndarray, known_correct: np.ndarray, unknown_score: np.ndarray,
                             n_points: int = 200) -> pd.DataFrame:
    """One row per τ (quantiles of the known scores): coverage, precision, false-match rate."""
    taus = np.quantile(known_score, np.linspace(0.0, 0.995, n_points))
    rows = []
    for tau in np.unique(taus):
        acc = known_score >= tau
        rows.append({
            "tau": float(tau),
            "coverage": float(acc.mean()),
            "precision": float(known_correct[acc].mean()) if acc.any() else np.nan,
            "false_match_rate": float((unknown_score >= tau).mean()),
        })
    return pd.DataFrame(rows).sort_values("coverage", ascending=False).reset_index(drop=True)


def tau_for_coverage(known_score: np.ndarray, coverage: float) -> float:
    """Smallest τ such that at least `coverage` of known queries are accepted."""
    return float(np.quantile(known_score, 1.0 - coverage))


def precision_at_coverage(curve: pd.DataFrame, coverage: float) -> float:
    """Precision of the curve point whose coverage is closest to (and >=) the target."""
    c = curve[curve.coverage >= coverage]
    row = c.iloc[c.coverage.argmin()] if len(c) else curve.iloc[0]
    return float(row.precision)


def apply_tau(known_score: np.ndarray, known_correct: np.ndarray, unknown_score: np.ndarray, tau: float) -> dict[str, float]:
    acc = known_score >= tau
    return {
        "tau": float(tau),
        "coverage": float(acc.mean()),
        "precision": float(known_correct[acc].mean()) if acc.any() else float("nan"),
        "false_match_rate": float((unknown_score >= tau).mean()),
        "unknown_rejected": float((unknown_score < tau).mean()),
    }
