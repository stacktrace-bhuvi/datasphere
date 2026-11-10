"""Paired statistical comparisons between diagnostics.

Rows that share an SCM are not independent (same data, several models/explainers/seeds),
so every test aggregates to one number per SCM first and then resamples or ranks SCMs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def per_scm_abs_error(preds: pd.DataFrame) -> pd.Series:
    """Mean absolute prediction error per SCM (index = scm_id)."""
    return (preds["y"] - preds["pred"]).abs().groupby(preds["scm_id"]).mean()


def _scm_index(preds: pd.DataFrame) -> tuple[list, list[np.ndarray]]:
    ids = sorted(preds["scm_id"].unique())
    pos = preds["scm_id"].to_numpy()
    return ids, [np.where(pos == s)[0] for s in ids]


def _spearman(y, p) -> float:
    if np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    v = spearmanr(y, p).statistic
    return float(v) if np.isfinite(v) else 0.0


def paired_bootstrap(preds_a: pd.DataFrame, preds_b: pd.DataFrame, n_boot: int = 2000,
                     seed: int = 0) -> dict[str, float]:
    """Bootstrap over SCMs of (MAE_b - MAE_a) and (Spearman_a - Spearman_b).

    Positive `mae_gain` / `spearman_gain` means method A is better than method B.
    Reports the mean gain, a 95% percentile CI and a two-sided bootstrap p-value.
    Rows are resampled SCM by SCM (cluster bootstrap), never individually.
    """
    ea, eb = per_scm_abs_error(preds_a), per_scm_abs_error(preds_b)
    ids = sorted(set(ea.index) & set(eb.index))
    ea, eb = ea.loc[ids].to_numpy(), eb.loc[ids].to_numpy()
    a = preds_a[preds_a["scm_id"].isin(ids)].sort_values(["scm_id"]).reset_index(drop=True)
    b = preds_b[preds_b["scm_id"].isin(ids)].sort_values(["scm_id"]).reset_index(drop=True)
    _, idx_a = _scm_index(a)
    _, idx_b = _scm_index(b)
    ya, pa, yb, pb = (a["y"].to_numpy(), a["pred"].to_numpy(), b["y"].to_numpy(), b["pred"].to_numpy())
    rng = np.random.default_rng(seed)
    n = len(ids)
    mae_gain = np.empty(n_boot)
    sp_gain = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, n, n)
        mae_gain[i] = eb[draw].mean() - ea[draw].mean()
        ra = np.concatenate([idx_a[j] for j in draw])
        rb = np.concatenate([idx_b[j] for j in draw])
        sp_gain[i] = _spearman(ya[ra], pa[ra]) - _spearman(yb[rb], pb[rb])

    def _summ(g):
        lo, hi = np.percentile(g, [2.5, 97.5])
        p = 2 * min((g <= 0).mean(), (g >= 0).mean())
        return float(g.mean()), float(lo), float(hi), float(min(1.0, p))

    m, mlo, mhi, mp = _summ(mae_gain)
    s, slo, shi, sp = _summ(sp_gain)
    return {"n_scms": n, "mae_gain": m, "mae_gain_ci_lo": mlo, "mae_gain_ci_hi": mhi, "mae_gain_p": mp,
            "spearman_gain": s, "spearman_gain_ci_lo": slo, "spearman_gain_ci_hi": shi, "spearman_gain_p": sp}


def wilcoxon_paired(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> dict[str, float]:
    """Wilcoxon signed-rank test on per-SCM absolute errors (A vs B)."""
    ea, eb = per_scm_abs_error(preds_a), per_scm_abs_error(preds_b)
    ids = sorted(set(ea.index) & set(eb.index))
    diff = eb.loc[ids].to_numpy() - ea.loc[ids].to_numpy()
    if len(ids) < 5 or np.allclose(diff, 0):
        return {"n_scms": len(ids), "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
                "frac_scms_a_better": float("nan")}
    res = wilcoxon(diff, alternative="two-sided", zero_method="zsplit")
    return {"n_scms": len(ids), "wilcoxon_stat": float(res.statistic), "wilcoxon_p": float(res.pvalue),
            "frac_scms_a_better": float((diff > 0).mean())}


def bootstrap_metric_ci(preds: pd.DataFrame, metric: str = "spearman", n_boot: int = 2000,
                        seed: int = 0) -> dict[str, float]:
    """Percentile bootstrap CI (cluster bootstrap over SCMs) of a pooled metric for one method."""
    p_ = preds.sort_values("scm_id").reset_index(drop=True)
    ids, idx = _scm_index(p_)
    y, p = p_["y"].to_numpy(), p_["pred"].to_numpy()
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([idx[j] for j in rng.integers(0, len(ids), len(ids))])
        yy, pp = y[rows], p[rows]
        if metric == "spearman":
            v = _spearman(yy, pp)
        elif metric == "mae":
            v = np.mean(np.abs(yy - pp))
        elif metric == "r2":
            v = 1 - np.sum((yy - pp) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-12)
        else:
            raise ValueError(metric)
        vals[i] = v if np.isfinite(v) else 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {f"{metric}_mean": float(vals.mean()), f"{metric}_ci_lo": float(lo), f"{metric}_ci_hi": float(hi)}
