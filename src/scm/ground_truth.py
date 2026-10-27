"""Causal ground truth computation and estimand gap evaluation per SCM."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.scm.scm import SCM
from src.scm.oracle import exact_shap


def compute_ground_truth(
    scm: SCM,
    df: pd.DataFrame,
    n_repeats: int = 5,
    n_ref: int = 50,
    n_noise_per_ref: int = 5_000,
    zero_tol: float = 1e-12,
    n_explain: int = 100,
    n_background: int = 50,
) -> dict[str, Any]:
    """Compute ground truth causal effects with Monte Carlo SE and oracle Shapley values.

    Parameters:
      scm: SCM instance.
      df: sampled DataFrame containing columns X1..Xd, Y, and row_split.
      n_repeats: number of random seed repeats for MC standard error.
      n_ref: reference points for nested true_effects.
      n_noise_per_ref: noise draws per reference point.
      zero_tol: threshold for zeroing oracle Shapley values.
      n_explain: number of fixed test explain instances (default 100).
      n_background: number of fixed train background instances (default 50).

    Returns:
      Dictionary adhering to ground_truth.json schema.
    """
    d = scm.d
    feature_cols = [f"X{i}" for i in range(1, d + 1)]

    # 1. Repeated true_effects estimation for MC mean and Standard Error
    tau_runs: list[np.ndarray] = []
    signed_tau_runs: list[np.ndarray] = []

    base_seed = scm.metadata.get("seed", 42)
    for r in range(n_repeats):
        rep_seed = base_seed + 10_000 * (r + 1)
        t_abs = scm.true_effects(
            delta_sd=1.0,
            seed=rep_seed,
            signed=False,
            n_ref=n_ref,
            n_noise_per_ref=n_noise_per_ref,
            mode="auto",
        )
        t_sgn = scm.true_effects(
            delta_sd=1.0,
            seed=rep_seed,
            signed=True,
            n_ref=n_ref,
            n_noise_per_ref=n_noise_per_ref,
            mode="auto",
        )
        tau_runs.append(t_abs)
        signed_tau_runs.append(t_sgn)

    tau_stack = np.vstack(tau_runs)  # (n_repeats, d)
    signed_stack = np.vstack(signed_tau_runs)

    tau_mean = np.mean(tau_stack, axis=0)
    tau_se = np.std(tau_stack, axis=0, ddof=1) / np.sqrt(n_repeats)

    signed_mean = np.mean(signed_stack, axis=0)
    signed_se = np.std(signed_stack, axis=0, ddof=1) / np.sqrt(n_repeats)

    # 2. Fixed explain and background sets from data splits
    test_mask = df["row_split"] == "test"
    train_mask = df["row_split"] == "train"

    test_indices = df[test_mask].index[:n_explain].tolist()
    train_indices = df[train_mask].index[:n_background].tolist()

    X_explain = df.loc[test_indices, feature_cols].to_numpy(dtype=np.float64)
    X_background = df.loc[train_indices, feature_cols].to_numpy(dtype=np.float64)

    # 3. Exact Oracle Shapley values on f_star
    phi = exact_shap(scm.f_star, X_explain, X_background, max_d=10, zero_tol=zero_tol)
    oracle_mean_abs_shap = np.mean(np.abs(phi), axis=0)

    # k = number of non-zero oracle features
    nonzero_mask = oracle_mean_abs_shap > 0.0
    k_oracle = int(np.sum(nonzero_mask))

    # 4. Estimand Gap Metrics
    l1_diff = np.sum(np.abs(tau_mean - oracle_mean_abs_shap))
    tau_l1 = np.sum(tau_mean)
    norm_l1 = float(l1_diff / (tau_l1 + 1e-12))

    l2_diff = float(np.linalg.norm(tau_mean - oracle_mean_abs_shap))
    tau_l2 = float(np.linalg.norm(tau_mean))
    norm_l2 = float(l2_diff / (tau_l2 + 1e-12))

    # Tie-aware Kendall tau-b
    kt_res = kendalltau(tau_mean, oracle_mean_abs_shap, variant="b")
    kt_stat = float(kt_res.statistic) if not np.isnan(kt_res.statistic) else 0.0

    # Top-k overlap where k = number of non-zero oracle features
    if k_oracle > 0:
        top_k_tau = set(np.argsort(-tau_mean)[:k_oracle])
        top_k_oracle = set(np.argsort(-oracle_mean_abs_shap)[:k_oracle])
        overlap_frac = float(len(top_k_tau.intersection(top_k_oracle)) / k_oracle)
    else:
        overlap_frac = 1.0

    # 5. Clipped fractions per node and for Y
    clip_bound = scm.metadata.get("clip_bound")
    clipped_fractions: dict[str, float] = {}
    if clip_bound is not None:
        threshold = float(clip_bound) - 1e-6
        for col in feature_cols + ["Y"]:
            vals = np.abs(df[col].to_numpy(dtype=np.float64))
            clipped_fractions[col] = float(np.mean(vals >= threshold))
    else:
        for col in feature_cols + ["Y"]:
            clipped_fractions[col] = 0.0

    return {
        "scm_id": scm.metadata.get("scm_id", ""),
        "d": d,
        "n_repeats": n_repeats,
        "tau": tau_mean.tolist(),
        "tau_se": tau_se.tolist(),
        "signed_tau": signed_mean.tolist(),
        "signed_tau_se": signed_se.tolist(),
        "explain_row_indices": test_indices,
        "background_row_indices": train_indices,
        "oracle_mean_abs_shap": oracle_mean_abs_shap.tolist(),
        "k_oracle": k_oracle,
        "estimand_gap": {
            "normalized_l1": norm_l1,
            "normalized_l2": norm_l2,
            "kendall_tau_b": kt_stat,
            "top_k_overlap": overlap_frac,
            "k": k_oracle,
        },
        "clipped_fractions": clipped_fractions,
        "clipped_fraction_Y": clipped_fractions.get("Y", 0.0),
    }
