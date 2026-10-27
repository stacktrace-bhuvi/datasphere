"""Exact interventional (marginal) Shapley value oracle."""

from __future__ import annotations
import math
from typing import Callable
import numpy as np


def _precompute_coalition_matrix(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute subset bitmasks and the Shapley coalition projection matrix W.

    Returns:
      masks: (2^d, d) boolean array where masks[s, j] is True iff feature j is in coalition s.
      W: (2^d, d) float64 matrix where:
           W[s, j] = + |s-1|! (d-|s|)! / d!  if j in S
           W[s, j] = - |s|! (d-|s|-1)! / d!  if j not in S
         Such that phi = v_table @ W computes all Shapley values via a single matrix product.
    """
    n_subsets = 1 << d
    masks = np.zeros((n_subsets, d), dtype=bool)
    W = np.zeros((n_subsets, d), dtype=np.float64)

    for s in range(n_subsets):
        for j in range(d):
            if (s >> j) & 1:
                masks[s, j] = True
        size = int(masks[s].sum())
        for j in range(d):
            if masks[s, j]:
                W[s, j] = (
                    math.factorial(size - 1)
                    * math.factorial(d - size)
                    / math.factorial(d)
                )
            else:
                W[s, j] = -(
                    math.factorial(size)
                    * math.factorial(d - size - 1)
                    / math.factorial(d)
                )

    return masks, W


def exact_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    max_d: int = 10,
    chunk_size: int = 8,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Compute exact interventional (marginal) Shapley values fully vectorized over coalitions.

    All 2^d coalitions are constructed, evaluated through f, and integrated via
    matrix multiplication in vectorized NumPy tensors without a Python loop over coalitions.

    Parameters:
      f: callable mapping (N, d) array to (N,) array of predictions.
      X_explain: (n_explain, d) array of instances to explain.
      X_background: (n_background, d) baseline/reference dataset.
      max_d: maximum allowed feature dimension (default 10).
      chunk_size: number of explain instances evaluated per chunk (default 8).
      zero_tol: values below this tolerance are zeroed to prevent floating-point noise from breaking ties.

    Returns:
      phi: (n_explain, d) array of exact Shapley values.
    """
    X_exp = np.asarray(X_explain, dtype=np.float64)
    X_bg = np.asarray(X_background, dtype=np.float64)

    if X_exp.ndim == 1:
        X_exp = X_exp.reshape(1, -1)
    if X_bg.ndim == 1:
        X_bg = X_bg.reshape(1, -1)

    n_explain, d = X_exp.shape
    n_bg, d_bg = X_bg.shape

    if d != d_bg:
        raise ValueError(f"Feature mismatch: X_explain has {d} features, X_background has {d_bg}")
    if d > max_d:
        raise ValueError(f"Feature dimension {d} exceeds max_d={max_d} for exact 2^d coalition enumeration.")

    masks, W = _precompute_coalition_matrix(d)
    n_subsets = 1 << d

    # Pre-broadcast background: shape (1, 1, n_bg, d)
    bg_bc = X_bg[np.newaxis, np.newaxis, :, :]
    # Pre-broadcast masks: shape (1, 2^d, 1, d)
    masks_bc = masks[np.newaxis, :, np.newaxis, :]

    phi_chunks: list[np.ndarray] = []

    # Process X_explain in chunks to maintain bounded memory
    for c_start in range(0, n_explain, chunk_size):
        c_end = min(c_start + chunk_size, n_explain)
        X_chunk = X_exp[c_start:c_end]
        cur_nc = len(X_chunk)

        # Broadcast explain points: shape (cur_nc, 1, 1, d)
        chunk_bc = X_chunk[:, np.newaxis, np.newaxis, :]

        # Construct all (cur_nc, 2^d, n_bg, d) hybrid combinations simultaneously
        hybrid = np.where(masks_bc, chunk_bc, bg_bc)

        # Vectorized evaluation over all coalitions and background draws
        flat_hybrid = hybrid.reshape(-1, d)
        preds = f(flat_hybrid).reshape(cur_nc, n_subsets, n_bg)

        # Marginalize over background: v_table shape (cur_nc, 2^d)
        v_table = np.mean(preds, axis=2)

        # Compute Shapley values via coalition projection matrix multiplication
        phi_chunk = v_table @ W
        phi_chunks.append(phi_chunk)

    phi_all = np.vstack(phi_chunks)
    if zero_tol is not None:
        phi_all[np.abs(phi_all) < zero_tol] = 0.0

    return phi_all


def mean_abs_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    n_explain: int = 100,
    n_background: int = 50,
    max_d: int = 10,
    chunk_size: int = 8,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Compute mean absolute Shapley values per feature across instances.

    Parameters:
      f: callable mapping (N, d) array to (N,) array of predictions.
      X_explain: array of instances to explain.
      X_background: array of baseline/reference instances.
      n_explain: maximum number of explain instances to evaluate (default 100).
      n_background: maximum number of background instances to evaluate (default 50).
      max_d: maximum allowed feature dimension (default 10).
      chunk_size: chunk size for vectorization (default 8).
      zero_tol: threshold below which attribution is set to exact 0.0.

    Returns:
      (d,) array containing (1 / N) * sum_i |phi_j(x_i)|.
    """
    X_exp = np.asarray(X_explain, dtype=np.float64)[:n_explain]
    X_bg = np.asarray(X_background, dtype=np.float64)[:n_background]

    phi = exact_shap(f, X_exp, X_bg, max_d=max_d, chunk_size=chunk_size, zero_tol=zero_tol)
    return np.mean(np.abs(phi), axis=0)
