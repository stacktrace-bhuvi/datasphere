"""Disagreement between explainers (and between models) on the same data.

If two explanation methods, or the same method on two different models, rank the
features differently, at least one of them is wrong about the data-generating
process. That is observable without ground truth, so it is a candidate error signal.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from src.metrics.error_metrics import importance_shares


def rank_disagreement(a, b) -> float:
    """1 - Kendall tau-b between two mean-|attribution| vectors (0 = same ranking, 2 = reversed)."""
    kt = kendalltau(np.asarray(a, float), np.asarray(b, float), variant="b").statistic
    kt = float(kt) if np.isfinite(kt) else 0.0
    return 1.0 - kt


def share_disagreement(a, b) -> float:
    """L2 distance between the importance shares of two vectors."""
    return float(np.linalg.norm(importance_shares(a) - importance_shares(b)))


def top_k_disagreement(a, b, k: int = 3) -> float:
    """1 - |top-k(a) & top-k(b)| / k."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    k = int(min(k, a.size))
    ta = set(np.argsort(-a, kind="stable")[:k])
    tb = set(np.argsort(-b, kind="stable")[:k])
    return 1.0 - len(ta & tb) / k


def explainer_disagreement(target: str, vectors: dict[str, np.ndarray],
                           reference: str = "lime") -> dict[str, float]:
    """Disagreement of explainer ``target`` with the others run on the same (scm, model, seed).

    vectors maps explainer name -> mean-|attribution| vector. Returns
      disagreement_rank_mean / disagreement_l2_mean   averaged over all OTHER explainers
      disagreement_rank_vs_<reference> / _l2_vs_<reference>  against one fixed reference
      disagreement_topk_mean                           mean top-3 disagreement
    Missing comparisons give NaN.
    """
    nan = float("nan")
    a = vectors[target]
    others = {k: v for k, v in vectors.items() if k != target}
    out = {
        "disagreement_rank_mean": nan, "disagreement_l2_mean": nan, "disagreement_topk_mean": nan,
        f"disagreement_rank_vs_{reference}": nan, f"disagreement_l2_vs_{reference}": nan,
    }
    if others:
        out["disagreement_rank_mean"] = float(np.mean([rank_disagreement(a, v) for v in others.values()]))
        out["disagreement_l2_mean"] = float(np.mean([share_disagreement(a, v) for v in others.values()]))
        out["disagreement_topk_mean"] = float(np.mean([top_k_disagreement(a, v) for v in others.values()]))
    ref = vectors.get(reference) if target != reference else None
    if target == reference and others:
        # the reference explainer compares itself to the mean of the others
        ref = np.mean(np.vstack(list(others.values())), axis=0)
    if ref is not None:
        out[f"disagreement_rank_vs_{reference}"] = rank_disagreement(a, ref)
        out[f"disagreement_l2_vs_{reference}"] = share_disagreement(a, ref)
    return out


def cross_model_disagreement(a, b) -> dict[str, float]:
    """Same explainer and seed, two different model families."""
    return {
        "cross_model_disagreement_rank": rank_disagreement(a, b),
        "cross_model_disagreement_l2": share_disagreement(a, b),
    }
