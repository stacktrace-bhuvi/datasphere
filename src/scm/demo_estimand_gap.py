"""Demonstration script comparing true interventional effects vs exact oracle SHAP.

Visualizes the structural Estimand Gap across 20 synthetic SCMs using
saved fixed explain (first 100 test rows) / background (first 50 train rows)
and tie-aware Kendall tau-b with top-k overlap where k = number of non-zero oracle features.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.scm.scm import SCM
from src.scm.oracle import mean_abs_shap
from src.scm.graph import GRAPH_FAMILIES


def compute_top_k_overlap(vec1: np.ndarray, vec2: np.ndarray, k: int) -> tuple[float, str]:
    """Compute fraction of top-k features in vec1 that also appear in top-k of vec2."""
    if k <= 0:
        return 1.0, "0/0"
    top_k_1 = set(np.argsort(-vec1)[:k])
    top_k_2 = set(np.argsort(-vec2)[:k])
    intersection = top_k_1.intersection(top_k_2)
    overlap_fraction = len(intersection) / float(k)
    overlap_str = f"{len(intersection)}/{k}"
    return overlap_fraction, overlap_str


def run_estimand_gap_demo(n_scms: int = 20) -> pd.DataFrame:
    """Generate n_scms SCMs, compute tau and mean|exact_shap(f*)|, and display comparison."""
    print("=" * 88)
    print("ESTIMAND GAP STUDY: True Interventional Effects vs. Exact Oracle SHAP of f*")
    print("=" * 88)

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

            # 2. Draw sample with 60/20/20 split for fixed explain/background sets
            obs = scm.sample(1500, seed=seed)
            n_samples = len(obs)
            n_train = int(0.60 * n_samples)
            n_val = int(0.20 * n_samples)
            n_test = n_samples - n_train - n_val

            splits_array = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
            split_rng = np.random.default_rng(seed + 999)
            split_rng.shuffle(splits_array)
            obs["row_split"] = splits_array

            feat_cols = [f"X{k}" for k in range(1, d + 1)]
            test_mask = obs["row_split"] == "test"
            train_mask = obs["row_split"] == "train"

            # Fixed sets: first 100 test rows for explain, first 50 train rows for background
            X_exp = obs[test_mask][feat_cols].iloc[:100].to_numpy(dtype=np.float64)
            X_bg = obs[train_mask][feat_cols].iloc[:50].to_numpy(dtype=np.float64)

            # 3. Exact SHAP on true noiseless function f_star
            shap_fstar = mean_abs_shap(scm.f_star, X_exp, X_bg, max_d=10, zero_tol=1e-12)

            # 4. Error metrics
            l1_gap = float(np.mean(np.abs(tau - shap_fstar)))
            max_gap = float(np.max(np.abs(tau - shap_fstar)))

            # Tie-aware Kendall tau-b correlation
            kt_res = kendalltau(tau, shap_fstar, variant="b")
            kt_stat = float(kt_res.statistic) if not np.isnan(kt_res.statistic) else 0.0

            # Number of non-zero oracle features k
            k_oracle = int(np.sum(shap_fstar > 0.0))
            overlap_frac, overlap_str = compute_top_k_overlap(tau, shap_fstar, k=k_oracle)

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
                "k (Oracle)": k_oracle,
                "Top-k Overlap": overlap_str,
                "Mediated Feats": mediated_nonparents_with_effect,
            })

    results_df = pd.DataFrame(rows)
    print("\nSummary of 20 SCMs:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 88)
    print("KEY FINDINGS:")
    print("1. In chain_mediation and layered SCMs, mediated features receive EXACT 0.0 SHAP")
    print("   attribution on f*, yet carry substantial true causal effects (tau > 0).")
    print("2. Top-k overlap is evaluated at k = count of non-zero oracle features.")
    print("3. Fixed explain (first 100 test rows) and background (first 50 train rows)")
    print("   establish an immutable baseline contract for model attribution benchmarks.")
    print("=" * 88 + "\n")

    return results_df


if __name__ == "__main__":
    run_estimand_gap_demo(n_scms=20)
