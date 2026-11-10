"""Diagnostic meta-models: predict explanation error from ground-truth-free observables."""
from __future__ import annotations

from typing import Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_MODEL_NAMES = ("gbm", "rf", "ridge")


def make_meta_model(name: str, seed: int = 0) -> Pipeline:
    """A sklearn pipeline: median imputation -> (scaling) -> regressor.

    gbm    gradient-boosted trees, shallow and regularised (our default diagnostic)
    rf     random forest
    ridge  standardised linear model (sanity check: is the signal linear?)
    """
    if name == "gbm":
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, min_samples_leaf=10, random_state=seed)
        steps = [("impute", SimpleImputer(strategy="median")), ("reg", reg)]
    elif name == "rf":
        reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                    random_state=seed, n_jobs=1)
        steps = [("impute", SimpleImputer(strategy="median")), ("reg", reg)]
    elif name == "ridge":
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                 ("reg", Ridge(alpha=1.0))]
    else:
        raise ValueError(f"unknown meta-model {name!r}; choose from {META_MODEL_NAMES}")
    return Pipeline(steps)


def meta_model_factory(name: str, seed: int = 0) -> Callable[[], Pipeline]:
    return lambda: make_meta_model(name, seed)
