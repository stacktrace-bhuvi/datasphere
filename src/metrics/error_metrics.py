"""Ranking and magnitude metrics for feature-attribution evaluation.

All inputs must use the same feature order and attribution scale. For causal
alignment, compare nonnegative magnitudes (e.g. SCM ``tau`` and mean absolute
SHAP); do not silently mix signed effects with absolute attributions.
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


def normalized_l2(reference, estimate, eps: float = 1e-12) -> float:
    """Return ||reference-estimate||_2 / (||reference||_2 + eps)."""
    a, b = _vector(reference, "reference"), _vector(estimate, "estimate")
    if a.shape != b.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


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
        **rank_metrics(tau, empirical_attribution),
    }
