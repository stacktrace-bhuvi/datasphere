"""Baseline diagnostics: what a practitioner could already do without a meta-model.

Each baseline is a small ridge regression (or a constant) on one conventional
warning signal. The meta-model has to beat these to be worth anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# name -> observable columns the baseline is allowed to see
BASELINE_FEATURES: dict[str, list[str]] = {
    "constant": [],                                   # predict the training mean
    "correlation_only": ["max_abs_corr", "mean_abs_corr"],
    "max_vif": ["max_vif"],
    "test_error": ["test_r2", "test_mse"],
    "sample_size": ["n_train", "d"],
    "disagreement_only": ["disagreement_rank_mean", "disagreement_l2_mean"],
    "stability_only": ["seed_stability_rank", "perturb_stability_rank"],
}


class ConstantPredictor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class BaselineDiagnostic:
    """Ridge on a fixed, small subset of observables (or the constant mean)."""

    def __init__(self, name: str):
        if name not in BASELINE_FEATURES:
            raise ValueError(f"unknown baseline {name!r}; choose from {list(BASELINE_FEATURES)}")
        self.name = name
        self.features = BASELINE_FEATURES[name]
        self.model = ConstantPredictor() if not self.features else Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0))])

    def _cols(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"baseline {self.name} needs columns {missing}")
        return X[self.features]

    def fit(self, X: pd.DataFrame, y):
        self.model.fit(self._cols(X) if self.features else X, np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._cols(X) if self.features else X), float)


def baseline_factory(name: str):
    return lambda: BaselineDiagnostic(name)
