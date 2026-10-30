"""Tests for src/models and src/explainers."""

import numpy as np
import pandas as pd
import pytest

from src.explainers import explain, load_attribution, save_attribution, supports
from src.models import TrainedModel, train_model
from src.scm.oracle import exact_shap
from src.scm.scm import SCM


@pytest.fixture(scope="module")
def dataset():
    scm = SCM.build("erdos_renyi_dag", 5, seed=3, nonlinearity_prob=0.3, interaction_prob=0.0)
    df = scm.sample(600, seed=3)
    df["row_split"] = ["train"] * 360 + ["val"] * 120 + ["test"] * 120
    feats = scm.features
    X_exp = df[df.row_split == "test"][feats].to_numpy()[:30]
    X_bg = df[df.row_split == "train"][feats].to_numpy()[:20]
    return scm, df, feats, X_exp, X_bg


@pytest.fixture(scope="module")
def xgb(dataset):
    return train_model("xgboost", dataset[1], seed=0, config={"n_estimators": 150})


class _Linear:
    """Stand-in estimator with known coefficients."""
    def __init__(self, beta):
        self.beta = np.asarray(beta, float)

    def predict(self, X):
        return np.asarray(X) @ self.beta


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("name", ["xgboost", "mlp"])
def test_models_learn_and_are_reproducible(dataset, name):
    df = dataset[1]
    cfg = {"n_estimators": 150} if name == "xgboost" else {"max_iter": 150}
    m1 = train_model(name, df, seed=0, config=cfg)
    m2 = train_model(name, df, seed=0, config=cfg)
    assert m1.metrics["test_r2"] > 0.2, m1.metrics
    assert m1.metrics["n_train"] == 360
    X = df[dataset[2]].to_numpy()[:50]
    assert m1.predict(X).shape == (50,)
    assert np.allclose(m1.predict(X), m2.predict(X))     # same seed -> same model


def test_unknown_model_rejected(dataset):
    with pytest.raises(ValueError):
        train_model("random_forest", dataset[1])


# ------------------------------------------------------------------ explainers
def test_exact_satisfies_efficiency(dataset, xgb):
    _, _, _, X_exp, X_bg = dataset
    a = explain(xgb, "exact", X_exp, X_bg)
    lhs = a.values.sum(axis=1)
    rhs = xgb.predict(X_exp) - xgb.predict(X_bg).mean()
    assert np.allclose(lhs, rhs, atol=1e-6)
    assert np.allclose(a.base_values + lhs, xgb.predict(X_exp), atol=1e-6)


def test_treeshap_matches_exact_shap(dataset, xgb):
    """Interventional TreeSHAP and brute-force exact SHAP compute the same quantity."""
    _, _, _, X_exp, X_bg = dataset
    tree = explain(xgb, "treeshap", X_exp, X_bg).values
    exact = explain(xgb, "exact", X_exp, X_bg).values
    assert np.allclose(tree, exact, atol=1e-4)


def test_kernelshap_approximates_exact(dataset, xgb):
    _, _, _, X_exp, X_bg = dataset
    k = explain(xgb, "kernelshap", X_exp, X_bg, seed=0).values
    e = explain(xgb, "exact", X_exp, X_bg).values
    assert np.corrcoef(k.ravel(), e.ravel())[0, 1] > 0.8


def test_kernelshap_budget_stays_below_full_enumeration():
    from src.explainers.attribution import kernel_budget
    for d in range(5, 11):
        assert 2 * d + 2 <= kernel_budget(d, "quarter") < 2 ** d


def test_kernelshap_is_not_silently_exact(dataset, xgb):
    _, _, _, X_exp, X_bg = dataset
    k = explain(xgb, "kernelshap", X_exp, X_bg, seed=0).values
    e = explain(xgb, "exact", X_exp, X_bg).values
    assert not np.allclose(k, e, atol=1e-6)


def test_lime_recovers_linear_contributions(dataset):
    _, df, feats, X_exp, X_bg = dataset
    beta = np.array([2.0, -1.0, 0.0, 0.5, 0.0])
    model = TrainedModel("linear", 0, _Linear(beta), feats, {})
    X_tr = df[df.row_split == "train"][feats].to_numpy()
    lime_vals = explain(model, "lime", X_exp[:10], X_bg, seed=0, X_train=X_tr).values
    expected = beta * (X_exp[:10] - X_tr.mean(axis=0))   # contribution of a linear model
    assert np.allclose(lime_vals, expected, atol=0.1)
    assert np.allclose(lime_vals[:, [2, 4]], 0.0, atol=0.05)   # unused features ~ 0


def test_permutation_is_global_and_ignores_unused_features(dataset):
    _, df, feats, X_exp, X_bg = dataset
    beta = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    model = TrainedModel("linear", 0, _Linear(beta), feats, {})
    te = df[df.row_split == "test"]
    a = explain(model, "permutation", X_exp, X_bg, X_test=te[feats].to_numpy(),
                y_test=te[feats].to_numpy() @ beta)
    assert a.is_global
    assert np.allclose(a.values, a.values[0])          # same vector on every row
    assert a.values[0, 0] > 1.0 and np.allclose(a.values[0, 1:], 0.0)


def test_treeshap_not_available_for_mlp(dataset):
    mlp = train_model("mlp", dataset[1], seed=0, config={"max_iter": 5})
    assert not supports(mlp, "treeshap")
    with pytest.raises(ValueError):
        explain(mlp, "treeshap", dataset[3], dataset[4])


# ------------------------------------------------------------------ files
def test_saved_files_follow_schema(tmp_path, dataset, xgb):
    _, _, _, X_exp, X_bg = dataset
    a = explain(xgb, "exact", X_exp, X_bg)
    rows = list(range(100, 100 + len(X_exp)))
    save_attribution(tmp_path, "scm_test", "xgboost", 0, a, rows, model_config=xgb.config)
    df, side = load_attribution(tmp_path, "scm_test", "xgboost", "exact", 0)
    assert list(df.columns) == ["row_index", "X1", "X2", "X3", "X4", "X5", "base_value"]
    assert pd.api.types.is_integer_dtype(df["row_index"]) and df["row_index"].tolist() == rows
    assert np.allclose(df[[f"X{j}" for j in range(1, 6)]].to_numpy(), a.values)
    for key in ["scm_id", "model", "explainer", "seed", "n_explained_rows",
                "mean_abs_attributions", "compute_time_seconds", "config", "is_global"]:
        assert key in side
    assert np.allclose(side["mean_abs_attributions"], np.abs(a.values).mean(axis=0))
