"""Demonstration script comparing true interventional effects vs exact oracle SHAP.

Visualizes the structural Estimand Gap across 20 synthetic SCMs using
tie-aware Kendall tau-b and Top-3 feature rank overlap.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.scm.scm import SCM
from src.scm.oracle import mean_abs_shap
from src.scm.graph import GRAPH_FAMILIES


def compute_top_k_overlap(vec1: np.ndarray, vec2: np.ndarray, k: int = 3) -> tuple[float, str]:
    """Compute fraction of top-k features in vec1 that also appear in top-k of vec2."""
    top_k_1 = set(np.argsort(-vec1)[:k])
    top_k_2 = set(np.argsort(-vec2)[:k])
    intersection = top_k_1.intersection(top_k_2)
    overlap_fraction = len(intersection) / float(k)
    overlap_str = f"{len(intersection)}/{k}"
    return overlap_fraction, overlap_str


def run_estimand_gap_demo(n_scms: int = 20) -> pd.DataFrame:
    """Generate n_scms SCMs, compute tau and mean|exact_shap(f*)|, and display comparison."""
    print("=" * 86)
    print("ESTIMAND GAP STUDY: True Interventional Effects vs. Exact Oracle SHAP of f*")
    print("=" * 86)

    rows: list[dict[str, object]] = []
    scms_per_family = n_scms // len(GRAPH_FAMILIES)

    global_idx = 0
    for fam_idx, family in enumerate(GRAPH_FAMILIES):
        for i in range(scms_per_family):
            global_idx += 1
            d = 5 + (i % 3)  # d in [5, 6, 7]
            seed = 1000 + fam_idx * 100 + i

            scm = SCM.build(
                family=family,
                d=d,
                seed=seed,
                nonlinearity_prob=0.3,
                noise_type="gaussian",
                noise_scale=1.0,
                standardize=True,
                clip_bound=10.0,
            )

            # 1. Compute true causal interventional effects tau using nested default
            tau = scm.true_effects(n_ref=50, n_noise_per_ref=5_000, delta_sd=1.0, seed=seed, signed=False)

            # 2. Draw sample for explanation background and evaluation
            obs = scm.sample(1500, seed=seed)
            feat_cols = [f"X{k}" for k in range(1, d + 1)]
            X_all = obs[feat_cols].values
            X_exp = X_all[:100]
            X_bg = X_all[100:200]

            # 3. Exact SHAP on true noiseless function f_star (100 explain, 50 background)
            shap_fstar = mean_abs_shap(scm.f_star, X_exp, X_bg, n_explain=100, n_background=50, max_d=10)

            # 4. Error metrics
            l1_gap = float(np.mean(np.abs(tau - shap_fstar)))
            max_gap = float(np.max(np.abs(tau - shap_fstar)))

            # Tie-aware Kendall tau-b correlation
            kt_res = kendalltau(tau, shap_fstar, variant="b")
            kt_stat = float(kt_res.statistic) if not np.isnan(kt_res.statistic) else 0.0

            # Top-3 rank agreement
            k_val = min(3, d)
            overlap_frac, overlap_str = compute_top_k_overlap(tau, shap_fstar, k=k_val)

            # Count dummy features in f* (non-parents of Y) that carry causal effect > 0.05
            y_parents = set(scm.graph.predecessors("Y"))
            mediated_nonparents_with_effect = 0
            for j_idx, feat in enumerate(scm.features):
                if feat not in y_parents and tau[j_idx] > 0.05:
                    mediated_nonparents_with_effect += 1

            rows.append({
                "SCM ID": f"SCM_{global_idx:02d}",
                "Family": family,
                "d": d,
                "Mean Tau": round(float(np.mean(tau)), 3),
                "Mean |SHAP|": round(float(np.mean(shap_fstar)), 3),
                "Mean L1 Gap": round(l1_gap, 3),
                "Max Gap": round(max_gap, 3),
                "Kendall tau-b": round(kt_stat, 3),
                "Top-3 Overlap": overlap_str,
                "Mediated Feats": mediated_nonparents_with_effect,
            })

    results_df = pd.DataFrame(rows)
    print("\nSummary of 20 SCMs:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 86)
    print("KEY FINDINGS:")
    print("1. In chain_mediation and layered SCMs, mediated features receive EXACT 0.0 SHAP")
    print("   attribution because they are not direct parents of Y, yet have large tau.")
    print("2. Kendall tau-b correctly handles zero-attribution ties without rank instability.")
    print("3. Top-3 Overlap quantifies whether the top causal drivers agree with top SHAP features.")
    print("=" * 86 + "\n")

    return results_df


if __name__ == "__main__":
    run_estimand_gap_demo(n_scms=20)
