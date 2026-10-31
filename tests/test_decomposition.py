import numpy as np
import pytest
from src.metrics import causal_alignment_metrics, decompose_error, normalized_l2, rank_metrics


def test_normalized_l2_zero_for_equal_vectors():
    assert normalized_l2([1, 2], [1, 2]) == 0.0


def test_rank_metrics_perfect_order():
    result = rank_metrics([1, 2, 3], [10, 20, 30])
    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["total_error_rank"] == pytest.approx(0.0)


def test_decomposition_telescopes_and_returns_schema_fields():
    tau = np.array([3.0, 2.0, 0.0])
    oracle = np.array([2.5, 1.0, 0.2])
    fitted = np.array([2.4, 1.1, 0.1])
    empirical = np.array([2.2, 1.2, 0.3])
    result = decompose_error(tau, oracle, fitted, empirical)

    assert np.allclose(
        result["total_error_vector"],
        np.array(result["estimand_gap_vector"])
        + np.array(result["model_misfit_vector"])
        + np.array(result["explainer_error_vector"])
    )
    for key in ["total_error_l2", "total_error_rank", "estimand_gap",
                "model_misfit", "explainer_error"]:
        assert key in result
        assert np.isfinite(result[key])


def test_decomposition_rejects_signed_or_negative_magnitude_input():
    with pytest.raises(ValueError, match="nonnegative"):
        decompose_error([1, -1], [1, 1], [1, 1], [1, 1])


def test_decomposition_rejects_different_dimensions():
    with pytest.raises(ValueError, match="same length"):
        decompose_error([1, 2], [1, 2], [1], [1, 2])


def test_causal_alignment_metrics():
    result = causal_alignment_metrics([3, 2, 0], [2.8, 2.2, 0.1])
    assert result["total_error_l2"] >= 0
    assert -1 <= result["kendall_tau_b"] <= 1
