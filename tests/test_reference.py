"""Tests for the fitted-model reference SHAP (Phi(f)) and its persistence in sidecars."""

import json

import numpy as np
import pytest

from src.explainers import (
    MissingReferenceError, add_reference_to_sidecar, attribution_from_reference,
    compute_reference_shap, explain, has_reference, load_attribution, read_reference,
    save_attribution,
)
from src.models import TrainedModel


class _Linear:
    def __init__(self, beta):
        self.beta = np.asarray(beta, float)

    def predict(self, X):
        return np.asarray(X) @ self.beta


class _Quad:
    """Non-additive model so SHAP differs from a plain linear closed form."""
    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1] + X[:, 2] ** 2


def _model(est, d):
    return TrainedModel("linear", 0, est, [f"X{j}" for j in range(1, d + 1)], {})


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    X_exp, X_bg = rng.normal(size=(12, 4)), rng.normal(size=(9, 4))
    return X_exp, X_bg, list(range(100, 112)), list(range(5, 14))


def test_reference_matches_closed_form_for_linear_model(data):
    X_exp, X_bg, ei, bi = data
    beta = np.array([2.0, -1.0, 0.0, 0.5])
    ref = compute_reference_shap(_model(_Linear(beta), 4), X_exp, X_bg, ei, bi)
    expected = np.mean(np.abs(beta * (X_exp - X_bg.mean(axis=0))), axis=0)
    assert np.allclose(ref.mean_abs, expected, atol=1e-10)
    assert ref.mean_abs[2] == 0.0                      # unused feature is exactly zero
    assert ref.method == "exact_shap_enumeration"


def test_reference_agrees_with_exact_explainer(data):
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    ref = compute_reference_shap(m, X_exp, X_bg, ei, bi)
    ex = explain(m, "exact", X_exp, X_bg)
    assert np.allclose(ref.values, ex.values, atol=1e-10)
    assert np.allclose(ref.mean_abs, ex.mean_abs, atol=1e-10)
    reused = attribution_from_reference(ref)
    assert reused.explainer == "exact" and np.allclose(reused.values, ex.values)
    reused.values[:] = 0                                # reuse must be a copy, not an alias
    assert not np.allclose(ref.values, 0)


def test_reference_is_not_the_empirical_explainer_output(data):
    """Reference stays exact even when the selected explainer is only approximate."""
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    ref = compute_reference_shap(m, X_exp, X_bg, ei, bi)
    lime = explain(m, "lime", X_exp, X_bg, seed=0, X_train=X_bg)
    assert not np.allclose(ref.mean_abs, lime.mean_abs, atol=1e-6)


def test_reference_sidecar_roundtrip_and_provenance(tmp_path, data):
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    ref = compute_reference_shap(m, X_exp, X_bg, ei, bi)
    attr = explain(m, "lime", X_exp, X_bg, seed=0, X_train=X_bg)
    save_attribution(tmp_path, "scm_t", "xgboost", 0, attr, ei, reference=ref)
    _, side = load_attribution(tmp_path, "scm_t", "xgboost", "lime", 0)
    vec, meta = read_reference(side, "t")
    assert np.allclose(vec, ref.mean_abs, rtol=0, atol=0)          # exact JSON round-trip
    assert not np.allclose(vec, side["mean_abs_attributions"], atol=1e-6)
    assert meta["explain_row_indices"] == ei and meta["background_row_indices"] == bi
    assert meta["feature_names"] == ["X1", "X2", "X3", "X4"]
    assert meta["n_explain"] == 12 and meta["n_background"] == 9
    assert "shared_with_exact_explainer" not in meta
    for key in ["scm_id", "mean_abs_attributions", "config", "is_global"]:   # old fields kept
        assert key in side


def test_exact_sidecar_is_flagged_as_sharing_the_reference(tmp_path, data):
    X_exp, X_bg, ei, bi = data
    ref = compute_reference_shap(_model(_Quad(), 4), X_exp, X_bg, ei, bi)
    save_attribution(tmp_path, "s", "xgboost", 0, attribution_from_reference(ref), ei, reference=ref)
    _, side = load_attribution(tmp_path, "s", "xgboost", "exact", 0)
    assert side["reference"]["shared_with_exact_explainer"] is True
    assert np.allclose(side["fitted_model_mean_abs_shap"], side["mean_abs_attributions"])


def test_save_rejects_reference_for_other_rows_or_dimension(tmp_path, data):
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    ref = compute_reference_shap(m, X_exp, X_bg, ei, bi)
    attr = explain(m, "exact", X_exp, X_bg)
    with pytest.raises(ValueError, match="different explain rows"):
        save_attribution(tmp_path, "s", "m", 0, attr, list(range(12)), reference=ref)
    ref3 = compute_reference_shap(_model(_Linear([1, 2, 3]), 3), X_exp[:, :3], X_bg[:, :3], ei, bi)
    with pytest.raises(ValueError, match="features"):
        save_attribution(tmp_path, "s", "m", 0, attr, ei, reference=ref3)


def test_legacy_sidecar_gets_actionable_error_not_a_fabricated_value(tmp_path, data):
    X_exp, X_bg, ei, _ = data
    attr = explain(_model(_Quad(), 4), "exact", X_exp, X_bg)
    save_attribution(tmp_path, "s", "m", 0, attr, ei)                 # legacy: no reference
    _, side = load_attribution(tmp_path, "s", "m", "exact", 0)
    assert "fitted_model_mean_abs_shap" not in side
    with pytest.raises(MissingReferenceError, match="run_attributions"):
        read_reference(side, "s__m__exact__seed0.json")


def test_backfill_adds_reference_and_leaves_explainer_output_untouched(tmp_path, data):
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    attr = explain(m, "lime", X_exp, X_bg, seed=0, X_train=X_bg)
    save_attribution(tmp_path, "s", "m", 0, attr, ei)
    path = tmp_path / "s__m__lime__seed0.json"
    before = json.loads(path.read_text())
    assert not has_reference(path)
    add_reference_to_sidecar(path, compute_reference_shap(m, X_exp, X_bg, ei, bi))
    after = json.loads(path.read_text())
    assert has_reference(path)
    for k, v in before.items():
        assert after[k] == v
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("bad", [
    {"fitted_model_mean_abs_shap": [0.1, -0.2]},                       # negative
    {"fitted_model_mean_abs_shap": [0.1, float("nan")]},               # non-finite
    {"fitted_model_mean_abs_shap": []},                                # empty
    {"fitted_model_mean_abs_shap": [0.1, 0.2], "reference": {"method": "kernelshap"}},  # wrong method
])
def test_read_reference_rejects_malformed(bad):
    side = {"reference": {"method": "exact_shap_enumeration"}, **bad}
    with pytest.raises(MissingReferenceError):
        read_reference(side, "x")


def test_reference_rejects_bad_inputs(data):
    X_exp, X_bg, ei, bi = data
    m = _model(_Quad(), 4)
    with pytest.raises(ValueError, match="same number of features"):
        compute_reference_shap(m, X_exp, X_bg[:, :3], ei, bi)
    with pytest.raises(ValueError, match="row-index"):
        compute_reference_shap(m, X_exp, X_bg, ei[:-1], bi)
    bad = X_exp.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_reference_shap(m, bad, X_bg, ei, bi)
    with pytest.raises(ValueError, match="features"):
        compute_reference_shap(_model(_Quad(), 3), X_exp, X_bg, ei, bi)


def test_reference_refuses_to_approximate_above_exact_limit():
    X = np.zeros((2, 11))
    with pytest.raises(ValueError, match="max_d"):
        compute_reference_shap(_model(_Linear(np.ones(11)), 11), X, X, [0, 1], [0, 1])


def test_nondeterministic_model_is_caught_by_efficiency_check(data):
    X_exp, X_bg, ei, bi = data

    class _Noisy:
        def predict(self, X):
            return np.asarray(X).sum(axis=1) + np.random.default_rng().normal(size=len(X))

    with pytest.raises(ArithmeticError, match="efficiency"):
        compute_reference_shap(_model(_Noisy(), 4), X_exp, X_bg, ei, bi)
