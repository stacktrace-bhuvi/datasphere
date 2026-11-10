"""Data-level observables: feature dependence, multicollinearity, target association.

Everything here is computed from the TRAIN split of one SCM's observational draws
only. Nothing uses the causal graph, tau, or the oracle, so these quantities are
available in any real deployment.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def _matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("X must be a 2-D array with at least 3 rows and 2 columns")
    return X


def abs_correlation_matrix(X) -> np.ndarray:
    """|Pearson correlation| between every pair of features, diagonal set to 0."""
    X = _matrix(X)
    sd = X.std(axis=0)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(np.abs(C), nan=0.0)
    C[np.arange(len(sd)), np.arange(len(sd))] = 0.0
    C[sd < 1e-12, :] = 0.0
    C[:, sd < 1e-12] = 0.0
    return C


def variance_inflation_factors(X) -> np.ndarray:
    """VIF_j = 1 / (1 - R_j^2) where R_j^2 is from regressing feature j on all others.

    Computed as the diagonal of the (pseudo-)inverse correlation matrix, which equals
    the per-feature VIF. Constant features get VIF 1.
    """
    X = _matrix(X)
    sd = X.std(axis=0)
    ok = sd > 1e-12
    vif = np.ones(X.shape[1])
    if ok.sum() >= 2:
        R = np.corrcoef(X[:, ok], rowvar=False)
        inv = np.linalg.pinv(R)
        vif[ok] = np.clip(np.diag(inv), 1.0, None)
    return vif


def mutual_information_matrix(X, seed: int = 0, n_neighbors: int = 3) -> np.ndarray:
    """Pairwise mutual information (nats) between features, k-NN estimator, diagonal 0."""
    X = _matrix(X)
    d = X.shape[1]
    M = np.zeros((d, d))
    for j in range(d):
        others = [i for i in range(d) if i > j]
        if not others:
            continue
        mi = mutual_info_regression(X[:, others], X[:, j], n_neighbors=n_neighbors,
                                    random_state=seed)
        for i, val in zip(others, mi):
            M[j, i] = M[i, j] = max(float(val), 0.0)
    return M


def _offdiag(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
    iu = np.triu_indices(d, k=1)
    return M[iu]


def data_observables(X_train, y_train, seed: int = 0) -> dict[str, float]:
    """Summary observables of one dataset's training split.

    Returns a flat dict with:
      max_abs_corr, mean_abs_corr        feature-feature |Pearson r|
      frac_corr_gt_0_3                   share of feature pairs with |r| > 0.3
      max_vif, mean_vif                  multicollinearity
      mutual_info_max, mutual_info_mean  feature-feature MI
      max_abs_corr_xy, mean_abs_corr_xy  feature-target |Pearson r|
      mi_xy_max, mi_xy_mean              feature-target MI
      n_train, d
    """
    X = _matrix(X_train)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if len(y) != len(X):
        raise ValueError("X_train and y_train must have the same number of rows")

    C = abs_correlation_matrix(X)
    off = _offdiag(C)
    vif = variance_inflation_factors(X)
    MI = mutual_information_matrix(X, seed=seed)
    mi_off = _offdiag(MI)

    sd = X.std(axis=0)
    r_xy = np.zeros(X.shape[1])
    if y.std() > 1e-12:
        for j in range(X.shape[1]):
            if sd[j] > 1e-12:
                r_xy[j] = abs(np.corrcoef(X[:, j], y)[0, 1])
    mi_xy = np.clip(mutual_info_regression(X, y, n_neighbors=3, random_state=seed), 0.0, None)

    return {
        "max_abs_corr": float(off.max()),
        "mean_abs_corr": float(off.mean()),
        "frac_corr_gt_0_3": float((off > 0.3).mean()),
        "max_vif": float(vif.max()),
        "mean_vif": float(vif.mean()),
        "mutual_info_max": float(mi_off.max()),
        "mutual_info_mean": float(mi_off.mean()),
        "max_abs_corr_xy": float(r_xy.max()),
        "mean_abs_corr_xy": float(r_xy.mean()),
        "mi_xy_max": float(mi_xy.max()),
        "mi_xy_mean": float(mi_xy.mean()),
        "n_train": int(len(X)),
        "d": int(X.shape[1]),
    }
