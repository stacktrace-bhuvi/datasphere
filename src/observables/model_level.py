"""Model-level observables: fit quality, over-fitting, prediction spread, cross-model agreement.

These use the fitted model and held-out labels, both of which exist in practice.
Nothing here touches the SCM or its ground truth.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def model_observables(model, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Observables of one trained model given {'train': (X, y), 'val': ..., 'test': ...}."""
    out: dict[str, float] = {}
    preds = {k: model.predict(X) for k, (X, _) in parts.items()}
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = parts["train"], parts["val"], parts["test"]

    out["test_mse"] = float(mean_squared_error(y_te, preds["test"]))
    out["test_r2"] = float(r2_score(y_te, preds["test"]))
    out["val_mse"] = float(mean_squared_error(y_va, preds["val"]))
    out["train_r2"] = float(r2_score(y_tr, preds["train"]))
    out["overfit_gap"] = float(out["train_r2"] - out["test_r2"])
    # How much of the outcome variance the model's predictions span (1 = matches y).
    out["pred_var_ratio"] = float(np.var(preds["test"]) / (np.var(y_te) + 1e-12))
    # Heavy-tailed residuals hint at a mis-specified model (missed non-linearity).
    resid = y_te - preds["test"]
    s = resid.std()
    out["resid_kurtosis"] = float(np.mean(((resid - resid.mean()) / (s + 1e-12)) ** 4) - 3.0) if s > 0 else 0.0
    return out


def cross_model_agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Pearson correlation between two models' predictions on the same rows (observable)."""
    a, b = np.asarray(pred_a, float).ravel(), np.asarray(pred_b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
