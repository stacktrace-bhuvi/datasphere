"""Three-component explanation-error decomposition.

The decomposition is performed in a common feature-wise magnitude space:
  tau_abs -> oracle_abs_shap -> fitted_model_abs_shap -> empirical_abs_attribution.

The vector differences telescope exactly. Their L2 norms do NOT generally add
up to the total L2 error; therefore component norms are reported separately
and must not be described as additive/orthogonal scalar errors.

Every component is reported twice:
  * ``*``        -- normalized L2 in the raw magnitude space (SCHEMAS.md section 3).
  * ``*_share``  -- L2 distance between importance shares (each vector divided by
                    its L1 norm). This is scale-free, so tau in outcome units and
                    SHAP in contribution units are compared on equal footing; it is
                    the headline magnitude metric from Review 1 onwards. Global
                    explainers such as permutation importance (MSE units) are only
                    meaningful in share or rank space.
"""
from __future__ import annotations
import numpy as np
from src.metrics.error_metrics import (
    importance_shares,
    normalized_l2,
    rank_metrics,
    top_k_overlap,
)


def _vec(x, name):
    a = np.asarray(x, dtype=float).reshape(-1)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any(a < 0):
        raise ValueError(f"{name} must be nonnegative magnitudes")
    return a


def decompose_error(tau_abs, oracle_abs_shap, fitted_model_abs_shap,
                    empirical_abs_attribution, eps: float = 1e-12) -> dict:
    """Decompose causal alignment gap into ordered vector differences.

    Args are feature-wise nonnegative magnitudes in identical feature order.
    ``oracle_abs_shap`` is exact SHAP for f*; ``fitted_model_abs_shap`` is
    exact/reference SHAP for fitted f; ``empirical_abs_attribution`` is the
    chosen approximation (sampled SHAP, LIME, etc.).

    Returns normalized L2 component magnitudes (raw and share space), total
    causal alignment error, rank error, top-k overlap, and raw component
    vectors for auditability.
    """
    names = ("tau_abs", "oracle_abs_shap", "fitted_model_abs_shap",
             "empirical_abs_attribution")
    vecs = [_vec(v, n) for v, n in zip(
        (tau_abs, oracle_abs_shap, fitted_model_abs_shap, empirical_abs_attribution), names
    )]
    if len({v.shape for v in vecs}) != 1:
        raise ValueError("All vectors must have the same length and feature order")
    tau, oracle, fitted, empirical = vecs

    estimand_vec = tau - oracle
    model_vec = oracle - fitted
    explainer_vec = fitted - empirical
    total_vec = tau - empirical

    # Verify the vector identity rather than incorrectly adding scalar norms.
    if not np.allclose(total_vec, estimand_vec + model_vec + explainer_vec,
                       rtol=1e-9, atol=1e-10):
        raise ArithmeticError("Error decomposition failed its telescoping identity")

    # Share space: the same telescoping identity holds for the share vectors.
    s_tau, s_oracle, s_fitted, s_emp = (importance_shares(v) for v in vecs)

    return {
        # --- raw magnitude space (SCHEMAS.md section 3)
        "total_error_l2": normalized_l2(tau, empirical, eps),
        "total_error_rank": rank_metrics(tau, empirical)["total_error_rank"],
        "estimand_gap": normalized_l2(tau, oracle, eps),
        "model_misfit": normalized_l2(oracle, fitted, eps),
        "explainer_error": normalized_l2(fitted, empirical, eps),
        # --- share space (scale-free)
        "total_error_share_l2": float(np.linalg.norm(s_tau - s_emp)),
        "estimand_gap_share": float(np.linalg.norm(s_tau - s_oracle)),
        "model_misfit_share": float(np.linalg.norm(s_oracle - s_fitted)),
        "explainer_error_share": float(np.linalg.norm(s_fitted - s_emp)),
        # --- rank-space components (1 - Kendall tau-b between consecutive stages)
        "estimand_gap_rank": rank_metrics(tau, oracle)["total_error_rank"],
        "model_misfit_rank": rank_metrics(oracle, fitted)["total_error_rank"],
        "explainer_error_rank": rank_metrics(fitted, empirical)["total_error_rank"],
        "top_k_overlap": top_k_overlap(tau, empirical),
        # --- audit vectors
        "estimand_gap_vector": estimand_vec.tolist(),
        "model_misfit_vector": model_vec.tolist(),
        "explainer_error_vector": explainer_vec.tolist(),
        "total_error_vector": total_vec.tolist(),
        "interpretation": (
            "Vector differences telescope exactly; normalized L2 component "
            "magnitudes are not additive. Inputs are absolute attribution/effect "
            "magnitudes, so this is a magnitude-space decomposition. The *_share "
            "fields repeat it on importance shares (scale-free)."
        ),
    }
