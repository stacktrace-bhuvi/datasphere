"""Run feature-attribution explainers on a trained model.

All explainers are evaluated on the SAME rows that the ground truth used:
    X_explain    = the 100 test rows in ground_truth.json["explain_row_indices"]
    X_background = the 50 train rows in ground_truth.json["background_row_indices"]
so every attribution is directly comparable to the oracle.

Each explainer returns an `Attribution`: a local (n_explain, d) matrix + base values.
Local attributions are in "contribution" units (how much feature j pushes f(x) away
from the baseline), which is the unit SHAP uses. LIME's coefficients are converted to
the same unit (coef * scaled feature value). Permutation importance is GLOBAL only, so
its single vector is repeated on every row and flagged with `is_global=True`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.train import TrainedModel
from src.scm.oracle import exact_shap

EXPLAINER_NAMES = ("exact", "treeshap", "kernelshap", "lime", "permutation")

DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "exact": {},
    "treeshap": {"feature_perturbation": "interventional"},
    # With d <= 10 there are only 2^d <= 1024 coalitions, so any budget >= 2^d makes
    # KernelSHAP enumerate everything and become exact. "quarter" = max(2d+2, 2^d / 4)
    # keeps it a genuine approximation at every d. An integer fixes the budget instead.
    "kernelshap": {"nsamples": "quarter"},
    "lime": {"num_samples": 1000, "kernel_width": None},
    "permutation": {"n_repeats": 10},
}


@dataclass
class Attribution:
    explainer: str
    values: np.ndarray            # (n_explain, d) local attributions
    base_values: np.ndarray       # (n_explain,)
    compute_time_seconds: float
    config: dict[str, Any] = field(default_factory=dict)
    is_global: bool = False

    @property
    def mean_abs(self) -> np.ndarray:
        return np.mean(np.abs(self.values), axis=0)


def supports(model: TrainedModel, explainer: str) -> bool:
    """TreeSHAP only works on tree models."""
    return not (explainer == "treeshap" and model.name != "xgboost")


def explain(model: TrainedModel, explainer: str, X_explain: np.ndarray, X_background: np.ndarray,
            seed: int = 0, config: dict[str, Any] | None = None,
            X_train: np.ndarray | None = None, X_test: np.ndarray | None = None,
            y_test: np.ndarray | None = None) -> Attribution:
    """Dispatch to one explainer. X_train is needed by LIME, (X_test, y_test) by permutation."""
    if explainer not in EXPLAINER_NAMES:
        raise ValueError(f"unknown explainer {explainer!r}; choose from {EXPLAINER_NAMES}")
    if not supports(model, explainer):
        raise ValueError(f"{explainer} is not available for model {model.name}")
    cfg = {**DEFAULT_CONFIGS[explainer], **(config or {})}
    X_explain = np.asarray(X_explain, dtype=np.float64)
    X_background = np.asarray(X_background, dtype=np.float64)

    t0 = time.perf_counter()
    fn = {"exact": _exact, "treeshap": _treeshap, "kernelshap": _kernelshap,
          "lime": _lime, "permutation": _permutation}[explainer]
    values, base, is_global = fn(model, X_explain, X_background, seed, cfg,
                                 X_train=X_train, X_test=X_test, y_test=y_test)
    return Attribution(explainer, np.asarray(values, dtype=np.float64),
                       np.asarray(base, dtype=np.float64).reshape(-1),
                       time.perf_counter() - t0, cfg, is_global)


# ------------------------------------------------------------------ explainers
def _exact(model, X_exp, X_bg, seed, cfg, **_):
    """Exact interventional SHAP of the FITTED model: Phi(f) in the error decomposition."""
    phi = exact_shap(model.predict, X_exp, X_bg)
    base = np.full(len(X_exp), model.predict(X_bg).mean())
    return phi, base, False


def _treeshap(model, X_exp, X_bg, seed, cfg, **_):
    import shap
    # "interventional" + the same background => estimates the same quantity as `exact`.
    ex = shap.TreeExplainer(model.estimator, data=X_bg,
                            feature_perturbation=cfg["feature_perturbation"])
    phi = ex.shap_values(X_exp, check_additivity=False)
    return phi, np.full(len(X_exp), float(np.ravel(ex.expected_value)[0])), False


def _kernelshap(model, X_exp, X_bg, seed, cfg, **_):
    import shap
    np.random.seed(seed)  # KernelExplainer samples coalitions with the global numpy RNG
    ex = shap.KernelExplainer(model.predict, X_bg, silent=True) if _kernel_has_silent() \
        else shap.KernelExplainer(model.predict, X_bg)
    phi = ex.shap_values(X_exp, nsamples=kernel_budget(X_exp.shape[1], cfg["nsamples"]), silent=True)
    return np.asarray(phi), np.full(len(X_exp), float(np.ravel(ex.expected_value)[0])), False


def kernel_budget(d: int, nsamples) -> int:
    """Number of coalitions KernelSHAP may evaluate."""
    if nsamples == "quarter":
        return int(max(2 * d + 2, 2 ** d // 4))
    return int(nsamples)


def _kernel_has_silent() -> bool:
    import inspect
    import shap
    return "silent" in inspect.signature(shap.KernelExplainer.__init__).parameters


def _lime(model, X_exp, X_bg, seed, cfg, X_train=None, **_):
    from lime.lime_tabular import LimeTabularExplainer
    if X_train is None:
        raise ValueError("LIME needs X_train (it samples perturbations from training statistics)")
    d = X_exp.shape[1]
    ex = LimeTabularExplainer(X_train, mode="regression", discretize_continuous=False,
                              kernel_width=cfg["kernel_width"], random_state=seed,
                              feature_names=[f"X{j + 1}" for j in range(d)])
    mean, scale = ex.scaler.mean_, ex.scaler.scale_
    vals, base = np.zeros_like(X_exp), np.zeros(len(X_exp))
    for i, x in enumerate(X_exp):
        e = ex.explain_instance(x, model.predict, num_features=d, num_samples=cfg["num_samples"])
        coef = np.zeros(d)
        for j, w in e.local_exp[1]:
            coef[int(j)] = w
        # coef is a slope per standard deviation; convert to a contribution like SHAP
        vals[i] = coef * (x - mean) / scale
        base[i] = float(np.ravel([e.intercept[1]])[0])
    return vals, base, False


def _permutation(model, X_exp, X_bg, seed, cfg, X_test=None, y_test=None, **_):
    """Global importance: increase in test MSE when feature j is shuffled."""
    from sklearn.inspection import permutation_importance
    if X_test is None or y_test is None:
        raise ValueError("permutation importance needs X_test and y_test")
    r = permutation_importance(_SkWrap(model), X_test, y_test, n_repeats=cfg["n_repeats"],
                               random_state=seed, scoring="neg_mean_squared_error")
    imp = np.asarray(r.importances_mean)
    vals = np.tile(imp, (len(X_exp), 1))
    return vals, np.full(len(X_exp), model.predict(X_bg).mean()), True


try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:  # pragma: no cover
    BaseEstimator = RegressorMixin = object


class _SkWrap(RegressorMixin, BaseEstimator):
    """Minimal sklearn-style wrapper so permutation_importance can call our model."""
    def __init__(self, model=None):
        self.model = model

    def fit(self, *a, **k):
        return self

    def predict(self, X):
        return self.model.predict(X)
