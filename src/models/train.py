"""Train the predictive models whose attributions we study (XGBoost and MLP).

Every model is trained on the `train` rows of data.parquet, uses the `val` rows only
for early stopping, and is scored on the `test` rows. The returned object always has
a fast, batched `.predict(X) -> (n,)` method, because the exact-SHAP oracle calls it
on up to ~5 million rows (100 explain x 50 background x 2^d coalitions).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

MODEL_NAMES = ("xgboost", "mlp")

DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "early_stopping_rounds": 30,
    },
    "mlp": {
        "hidden_layer_sizes": (64, 64),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        "n_iter_no_change": 20,
    },
}


@dataclass
class TrainedModel:
    """A fitted model plus everything downstream modules need to know about it."""

    name: str
    seed: int
    estimator: Any
    feature_names: list[str]
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.asarray(self.estimator.predict(X), dtype=np.float64).ravel()


def split_xy(df: pd.DataFrame, feature_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'train': (X, y), 'val': (X, y), 'test': (X, y)} from a data.parquet frame."""
    out = {}
    for part in ("train", "val", "test"):
        rows = df[df["row_split"] == part]
        out[part] = (rows[feature_names].to_numpy(np.float64), rows["Y"].to_numpy(np.float64))
    return out


def train_model(name: str, df: pd.DataFrame, seed: int = 0,
                config: dict[str, Any] | None = None) -> TrainedModel:
    """Train one model on one SCM dataset and report its test error."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    feature_names = [c for c in df.columns if c.startswith("X")]
    parts = split_xy(df, feature_names)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = parts["train"], parts["val"], parts["test"]
    cfg = {**DEFAULT_CONFIGS[name], **(config or {})}

    if name == "xgboost":
        est = XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", **cfg)
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        # Train with early stopping judged on OUR validation rows (not sklearn's
        # internal random split), so all models use the same data partition.
        est = MLPRegressor(random_state=seed, early_stopping=False, warm_start=True,
                           **{**cfg, "max_iter": 1})
        best, best_state, bad = np.inf, None, 0
        for _ in range(int(cfg["max_iter"])):
            with warnings.catch_warnings():          # 1-epoch fits always "fail to converge"
                warnings.simplefilter("ignore", ConvergenceWarning)
                est.fit(X_tr, y_tr)                  # one more epoch (warm start)
            val = mean_squared_error(y_va, est.predict(X_va))
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = ([w.copy() for w in est.coefs_], [b.copy() for b in est.intercepts_])
            else:
                bad += 1
                if bad >= int(cfg["n_iter_no_change"]):
                    break
        est.coefs_, est.intercepts_ = best_state

    model = TrainedModel(name, seed, est, feature_names, cfg)
    pred_te = model.predict(X_te)
    model.metrics = {
        "test_mse": float(mean_squared_error(y_te, pred_te)),
        "test_r2": float(r2_score(y_te, pred_te)),
        "val_mse": float(mean_squared_error(y_va, model.predict(X_va))),
        "n_train": int(len(y_tr)),
    }
    return model
