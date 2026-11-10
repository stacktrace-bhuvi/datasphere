"""Evaluation protocols for diagnostic models.

  grouped_cv          StratifiedGroupKFold over SCMs (groups = scm_id, strata = family):
                      in-distribution generalisation to UNSEEN SCMs from SEEN families.
  logfo               Leave-One-Graph-Family-Out: train on 4 families, test on the 5th.
                      Repeated for every family; this is the headline protocol.
  designated_split    manifest split: train families -> held-out family (layered).

Every protocol returns out-of-sample predictions for every row, so downstream
statistics (paired bootstrap, Wilcoxon) can be computed per SCM.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.diagnostic.dataset import design_matrix


def score(y_true, y_pred, high_threshold: float | None = None) -> dict[str, float]:
    """R^2, Spearman, MAE, RMSE and AUROC for flagging high-error rows (y > threshold)."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12 else 0.0,
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "n": int(len(y)),
    }
    if high_threshold is not None:
        flag = (y > high_threshold).astype(int)
        out["auroc_high_error"] = float(roc_auc_score(flag, p)) if 0 < flag.sum() < len(flag) else float("nan")
    return out


def _fit_predict(factory: Callable, X_tr, y_tr, X_te) -> np.ndarray:
    model = factory()
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), float)


def grouped_cv(df: pd.DataFrame, target: str, factory: Callable, features: list[str] | None = None,
               n_splits: int = 5, seed: int = 0, include_method_indicators: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions with groups = scm_id and strata = family."""
    X, _ = design_matrix(df, features, include_method_indicators)
    y = df[target].to_numpy(float)
    groups = df["scm_id"].to_numpy()
    strata = df["family"].astype("category").cat.codes.to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    thr = np.full(len(df), np.nan)
    for k, (tr, te) in enumerate(cv.split(X, strata, groups)):
        pred[te] = _fit_predict(factory, X.iloc[tr], y[tr], X.iloc[te])
        fold[te] = k
        thr[te] = np.median(y[tr])
    return pd.DataFrame({"scm_id": df["scm_id"], "family": df["family"], "model": df["model"],
                         "explainer": df["explainer"], "seed": df["seed"], "y": y, "pred": pred,
                         "fold": fold, "train_median": thr})


def logfo(df: pd.DataFrame, target: str, factory: Callable, features: list[str] | None = None,
          include_method_indicators: bool = True) -> pd.DataFrame:
    """Leave-one-graph-family-out predictions; `fold` holds the held-out family name."""
    X, _ = design_matrix(df, features, include_method_indicators)
    y = df[target].to_numpy(float)
    fam = df["family"].to_numpy()
    pred = np.full(len(df), np.nan)
    thr = np.full(len(df), np.nan)
    for f in sorted(set(fam)):
        te, tr = np.where(fam == f)[0], np.where(fam != f)[0]
        pred[te] = _fit_predict(factory, X.iloc[tr], y[tr], X.iloc[te])
        thr[te] = np.median(y[tr])
    return pd.DataFrame({"scm_id": df["scm_id"], "family": df["family"], "model": df["model"],
                         "explainer": df["explainer"], "seed": df["seed"], "y": y, "pred": pred,
                         "fold": fam, "train_median": thr})


def designated_split(df: pd.DataFrame, target: str, factory: Callable, features: list[str] | None = None,
                     include_method_indicators: bool = True) -> pd.DataFrame:
    """Train on manifest split == 'train', predict on split == 'heldout'."""
    X, _ = design_matrix(df, features, include_method_indicators)
    y = df[target].to_numpy(float)
    tr = np.where(df["split"].to_numpy() == "train")[0]
    te = np.where(df["split"].to_numpy() == "heldout")[0]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("designated_split needs both 'train' and 'heldout' rows")
    pred = _fit_predict(factory, X.iloc[tr], y[tr], X.iloc[te])
    sub = df.iloc[te]
    return pd.DataFrame({"scm_id": sub["scm_id"], "family": sub["family"], "model": sub["model"],
                         "explainer": sub["explainer"], "seed": sub["seed"], "y": y[te], "pred": pred,
                         "fold": "heldout", "train_median": np.median(y[tr])})


PROTOCOLS = {"grouped_cv": grouped_cv, "logfo": logfo, "designated_split": designated_split}


def summarize_predictions(preds: pd.DataFrame, by_fold: bool = True) -> pd.DataFrame:
    """Score pooled predictions and, optionally, each fold (family) separately."""
    rows = [{"fold": "pooled", **score(preds["y"], preds["pred"],
                                        high_threshold=float(np.nanmedian(preds["train_median"])))}]
    if by_fold:
        for f, g in preds.groupby("fold", sort=True):
            rows.append({"fold": str(f), **score(g["y"], g["pred"],
                                                 high_threshold=float(g["train_median"].iloc[0]))})
    return pd.DataFrame(rows)


def summarize_by_explainer(preds: pd.DataFrame) -> pd.DataFrame:
    """Score pooled predictions within each (model, explainer): can the diagnostic rank SCMs for a FIXED method?"""
    rows = []
    for (m, e), g in preds.groupby(["model", "explainer"], sort=True):
        rows.append({"model": m, "explainer": e,
                     **score(g["y"], g["pred"], high_threshold=float(np.nanmedian(g["train_median"])))})
    return pd.DataFrame(rows)


def run_methods(df: pd.DataFrame, target: str, methods: dict[str, Callable], protocol: str,
                features: list[str] | None = None, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every method under one protocol. Returns (scores, long predictions).

    The scores frame holds one row per fold plus a pooled row, and one row per
    (model, explainer) subset (fold = "by_method:<model>/<explainer>").
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    all_scores, all_preds = [], []
    for name, factory in methods.items():
        preds = PROTOCOLS[protocol](df, target, factory, features=features, **kw)
        preds.insert(0, "method", name)
        s = summarize_predictions(preds)
        by = summarize_by_explainer(preds)
        by.insert(0, "fold", "by_method:" + by["model"] + "/" + by["explainer"])
        by = by.drop(columns=["model", "explainer"])
        s = pd.concat([s, by], ignore_index=True)
        s.insert(0, "method", name)
        s.insert(0, "protocol", protocol)
        s.insert(0, "target", target)
        all_scores.append(s)
        all_preds.append(preds)
    return pd.concat(all_scores, ignore_index=True), pd.concat(all_preds, ignore_index=True)
