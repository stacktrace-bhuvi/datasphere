"""Edge cases for src/metrics/error_metrics.py and decomposition input handling."""
import warnings

import numpy as np
import pytest
from scipy.stats import kendalltau

from src.metrics import decompose_error, normalized_l2, rank_metrics


def test_normalized_l2_matches_formula():
    a, b = np.array([3.0, 4.0]), np.array([0.0, 4.0])
    assert normalized_l2(a, b) == pytest.approx(3.0 / 5.0)
    assert normalized_l2(a, a * 0) == pytest.approx(1.0)          # predicting zeros -> error 1


def test_normalized_l2_is_asymmetric_in_the_denominator():
    a, b = np.array([1.0, 0.0]), np.array([3.0, 0.0])
    assert normalized_l2(a, b) == pytest.approx(2.0)
    assert normalized_l2(b, a) == pytest.approx(2.0 / 3.0)


def test_zero_reference_uses_eps_denominator_and_stays_finite():
    v = normalized_l2([0.0, 0.0], [1.0, 0.0])
    assert np.isfinite(v) and v == pytest.approx(1.0 / 1e-12)    # documented SCHEMAS.md formula
    assert normalized_l2([0.0, 0.0], [0.0, 0.0]) == 0.0


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, np.inf], [], [[1.0, 2.0], [3.0, 4.0]]])
def test_invalid_vectors_rejected(bad):
    with pytest.raises(ValueError):
        normalized_l2([1.0, 2.0], bad)
    with pytest.raises(ValueError):
        rank_metrics(bad, [1.0, 2.0])


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="same shape"):
        normalized_l2([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="same shape"):
        rank_metrics([1, 2, 3], [1, 2])


def test_2d_array_is_not_silently_flattened_to_match_a_vector():
    with pytest.raises(ValueError, match="1-D"):
        normalized_l2(np.ones((2, 2)), np.ones(4))
    with pytest.raises(ValueError, match="1-D"):
        decompose_error(np.ones((2, 2)), np.ones(4), np.ones(4), np.ones(4))


def test_kendall_tau_b_handles_ties_like_scipy():
    tau = [0.0, 0.0, 0.5, 1.0, 0.0]
    est = [0.0, 0.001, 0.4, 2.0, 0.0]
    got = rank_metrics(tau, est)["kendall_tau_b"]
    assert got == pytest.approx(kendalltau(tau, est, variant="b").statistic)
    assert rank_metrics([0, 0, 1], [0, 0, 2])["kendall_tau_b"] == pytest.approx(1.0)


def test_reversed_order_gives_maximal_rank_error():
    r = rank_metrics([1, 2, 3, 4], [4, 3, 2, 1])
    assert r["kendall_tau_b"] == pytest.approx(-1.0) and r["total_error_rank"] == pytest.approx(2.0)
    assert r["spearman_rho"] == pytest.approx(-1.0)


def test_constant_vector_gives_no_rank_information_not_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rank_metrics([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert r["kendall_tau_b"] == 0.0 and r["spearman_rho"] == 0.0 and r["total_error_rank"] == 1.0


def test_rank_metrics_are_scale_invariant_but_l2_is_not():
    tau, est = np.array([3.0, 2.0, 1.0]), np.array([1.1, 0.9, 0.2])
    assert rank_metrics(tau, est)["total_error_rank"] == rank_metrics(tau, est * 100)["total_error_rank"]
    assert normalized_l2(tau, est) != pytest.approx(normalized_l2(tau, est * 100))


def test_decomposition_vectors_telescope_but_component_norms_do_not_add():
    tau, oracle = np.array([3.0, 2.0, 0.0]), np.array([2.5, 1.0, 0.2])
    fitted, emp = np.array([2.4, 1.1, 0.1]), np.array([2.2, 1.2, 0.3])
    r = decompose_error(tau, oracle, fitted, emp)
    parts = (np.array(r["estimand_gap_vector"]) + np.array(r["model_misfit_vector"])
             + np.array(r["explainer_error_vector"]))
    assert np.allclose(parts, r["total_error_vector"])
    scalar_sum = r["estimand_gap"] + r["model_misfit"] + r["explainer_error"]
    assert abs(scalar_sum - r["total_error_l2"]) > 1e-3          # norms with different denominators
    assert "not additive" in r["interpretation"]


def test_decomposition_with_identical_chain_is_zero():
    v = [1.0, 2.0, 0.0]
    r = decompose_error(v, v, v, v)
    assert r["total_error_l2"] == r["estimand_gap"] == r["model_misfit"] == r["explainer_error"] == 0.0
    assert r["total_error_rank"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_decomposition_rejects_nonfinite(bad):
    with pytest.raises(ValueError, match="finite"):
        decompose_error([1, bad], [1, 1], [1, 1], [1, 1])
