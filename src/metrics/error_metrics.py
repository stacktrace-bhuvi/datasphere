"""Ranking and magnitude metrics for feature-attribution evaluation.

All inputs must use the same feature order and attribution scale. For causal
alignment, compare nonnegative magnitudes (e.g. SCM ``tau`` and mean absolute
SHAP); do not silently mix signed effects with absolute attributions.

Two magnitude spaces are supported:
  * raw      -- vectors as given (tau in outcome units, SHAP in contribution units).
  * share    -- each vector divided by its L1 norm so it sums to one. This is the
                scale-free comparison recommended at Review 1: tau and SHAP live on
                different scales, so a raw L2 error never reaches zero even when the
                explanation ranks and proportions features perfectly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def importance_shares(values, eps: float = 1e-12) -> np.ndarray:
    """Return |values| / sum(|values|). An all-zero vector maps to the uniform share."""
    a = np.abs(_vector(values, "values"))
    total = float(a.sum())
    if total <= eps:
        return np.full_like(a, 1.0 / a.size)
    return a / total


def normalized_l2(reference, estimate, eps: float = 1e-12) -> float:
    """Return ||reference-estimate||_2 / (||reference||_2 + eps)."""
    a, b = _vector(reference, "reference"), _vector(estimate, "estimate")
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def share_l2(reference, estimate) -> float:
    """L2 distance between the importance shares of two vectors (scale-free, in [0, sqrt(2)])."""
    a, b = importance_shares(reference), importance_shares(estimate)
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.linalg.norm(a - b))


def top_k_overlap(reference, estimate, k: int | None = None) -> float:
    """Fraction of the top-k reference features that are also top-k in the estimate.

    k defaults to the number of strictly positive reference entries (the SCM's
    causally active features), capped at the vector length.
    """
    a, b = _vector(reference, "reference"), _vector(estimate, "estimate")
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have the same shape")
    if k is None:
        k = int(np.sum(a > 0))
    k = int(min(max(k, 0), a.size))
    if k == 0:
        return 1.0
    top_a = set(np.argsort(-a, kind="stable")[:k])
    top_b = set(np.argsort(-np.abs(b), kind="stable")[:k])
    return float(len(top_a & top_b) / k)


def rank_metrics(reference, estimate) -> dict[str, float]:
    """Return Kendall tau-b, Spearman rho, and 1 - Kendall tau-b rank error."""
    a, b = _vector(reference, "reference"), _vector(estimate, "estimate")
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have the same shape")
    kt = kendalltau(a, b, variant="b").statistic
    sp = spearmanr(a, b).statistic
    kt = float(kt) if np.isfinite(kt) else 0.0
    sp = float(sp) if np.isfinite(sp) else 0.0
    return {
        "kendall_tau_b": kt,
        "spearman_rho": sp,
        "total_error_rank": float(1.0 - kt),
    }


def causal_alignment_metrics(tau, empirical_attribution) -> dict[str, float]:
    """Compare causal-effect magnitudes with empirical attribution magnitudes."""
    return {
        "total_error_l2": normalized_l2(tau, empirical_attribution),
        "total_error_share_l2": share_l2(tau, empirical_attribution),
        "top_k_overlap": top_k_overlap(tau, empirical_attribution),
        **rank_metrics(tau, empirical_attribution),
    }
