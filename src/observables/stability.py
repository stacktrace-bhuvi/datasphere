"""Explainer-level observables: concentration, seed stability, perturbation stability, sensitivity.

All functions work on attribution matrices of shape (n_explain, d) or on the
per-feature mean-absolute vector of shape (d,). None of them need ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from src.metrics.error_metrics import importance_shares


def _kendall(a, b) -> float:
    kt = kendalltau(np.asarray(a, float), np.asarray(b, float), variant="b").statistic
    return float(kt) if np.isfinite(kt) else 0.0


def gini(values) -> float:
    """Gini coefficient of nonnegative values (0 = uniform, -> 1 = one feature takes all)."""
    v = np.sort(np.abs(np.asarray(values, float)).ravel())
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def concentration_observables(values: np.ndarray) -> dict[str, float]:
    """Concentration of a local attribution matrix (n_explain, d).

    attribution_gini   Gini of the mean-|attribution| shares across features
    top1_share         share of the most important feature
    effective_features exp(entropy of shares): how many features "really" get credit
    sign_consistency   share-weighted fraction of rows whose local sign matches the
                       feature's majority sign (1 = monotone explanation)
    local_dispersion   mean over features of std(local attr) / (mean |attr| + eps)
    """
    V = np.asarray(values, float)
    if V.ndim != 2:
        raise ValueError("values must be (n_explain, d)")
    mean_abs = np.abs(V).mean(axis=0)
    shares = importance_shares(mean_abs)
    ent = -np.sum(shares[shares > 0] * np.log(shares[shares > 0]))

    sign = np.sign(V)
    majority = np.where(sign.sum(axis=0) >= 0, 1.0, -1.0)
    agree = (sign == majority).mean(axis=0)
    agree[np.all(sign == 0, axis=0)] = 1.0

    disp = V.std(axis=0) / (mean_abs + 1e-12)
    return {
        "attribution_gini": gini(mean_abs),
        "top1_share": float(shares.max()),
        "effective_features": float(np.exp(ent)),
        "sign_consistency": float(np.sum(shares * agree)),
        "local_dispersion": float(np.mean(disp)),
    }


def seed_stability(mean_abs_vectors: list[np.ndarray]) -> dict[str, float]:
    """Agreement of an explainer with itself across random seeds (same SCM, model).

    seed_stability_rank   mean pairwise Kendall tau-b of the mean-|attr| vectors (1 = identical ranking)
    seed_stability_l2     mean pairwise L2 distance between importance shares (0 = identical)
    Returns NaN for both when fewer than two seeds are available.
    """
    vecs = [np.asarray(v, float).ravel() for v in mean_abs_vectors]
    if len(vecs) < 2:
        return {"seed_stability_rank": float("nan"), "seed_stability_l2": float("nan")}
    kts, l2s = [], []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            kts.append(_kendall(vecs[i], vecs[j]))
            l2s.append(float(np.linalg.norm(importance_shares(vecs[i]) - importance_shares(vecs[j]))))
    return {"seed_stability_rank": float(np.mean(kts)), "seed_stability_l2": float(np.mean(l2s))}


def perturbation_observables(values: np.ndarray, values_perturbed: np.ndarray,
                             X: np.ndarray, X_perturbed: np.ndarray) -> dict[str, float]:
    """Compare attributions before/after a small Gaussian input perturbation.

    perturb_stability_rank  mean over rows of Kendall tau-b(local attr, perturbed local attr)
    perturb_rel_change      mean over rows of ||delta attr|| / (||attr|| + eps)
    sensitivity             max over rows of ||delta attr|| / ||delta x|| (Lipschitz estimate)
    global_share_shift      L2 distance between the mean-|attr| shares before and after
    """
    A, B = np.asarray(values, float), np.asarray(values_perturbed, float)
    X0, X1 = np.asarray(X, float), np.asarray(X_perturbed, float)
    n = min(len(A), len(B))
    A, B, X0, X1 = A[:n], B[:n], X0[:n], X1[:n]
    d_attr = np.linalg.norm(A - B, axis=1)
    d_x = np.linalg.norm(X0 - X1, axis=1)
    kts = [_kendall(np.abs(A[i]), np.abs(B[i])) for i in range(n)]
    return {
        "perturb_stability_rank": float(np.mean(kts)) if n else float("nan"),
        "perturb_rel_change": float(np.mean(d_attr / (np.linalg.norm(A, axis=1) + 1e-12))) if n else float("nan"),
        "sensitivity": float(np.max(d_attr / (d_x + 1e-12))) if n else float("nan"),
        "global_share_shift": float(np.linalg.norm(importance_shares(np.abs(A).mean(0))
                                                   - importance_shares(np.abs(B).mean(0)))) if n else float("nan"),
    }


def perturb_inputs(X: np.ndarray, X_reference: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add N(0, (sigma * sd_j)^2) noise to every column, sd_j taken from X_reference."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    sd = np.asarray(X_reference, float).std(axis=0)
    sd[sd < 1e-12] = 1.0
    return X + rng.normal(0.0, 1.0, size=X.shape) * (sigma * sd)
