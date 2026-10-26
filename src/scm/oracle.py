"""Exact interventional (marginal) Shapley value oracle."""

from __future__ import annotations
import math
from itertools import combinations
from typing import Callable
import numpy as np


def _precompute_coalitions(d: int) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[list[int]]]:
    """Precompute subset bitmasks, weights, and membership lists for d features.

    Returns:
      masks: boolean array of shape (2^d, d) where masks[s, j] is True if j in S.
      weights: float array of length 2^d with Shapley weights w(|S|).
      subsets_with_j: for each feature j, list of subset indices where j in S.
      subsets_without_j: for each feature j, list of subset indices where j not in S.
    """
    n_subsets = 1 << d
    masks = np.zeros((n_subsets, d), dtype=bool)
    weights = np.zeros(n_subsets, dtype=np.float64)

    for s_idx in range(n_subsets):
        for j in range(d):
            if (s_idx >> j) & 1:
                masks[s_idx, j] = True
        s_size = int(masks[s_idx].sum())
        # Shapley weight for subset S of size |S| when adding feature j (out of d):
        # |S|! * (d - |S| - 1)! / d!
        if s_size < d:
            weights[s_idx] = (
                math.factorial(s_size) * math.factorial(d - s_size - 1) / math.factorial(d)
            )

    subsets_with_j: list[list[int]] = [[] for _ in range(d)]
    subsets_without_j: list[list[int]] = [[] for _ in range(d)]

    for s_idx in range(n_subsets):
        for j in range(d):
            if masks[s_idx, j]:
                subsets_with_j[j].append(s_idx)
            else:
                subsets_without_j[j].append(s_idx)

    return masks, weights, subsets_with_j, subsets_without_j


def exact_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    max_d: int = 10,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute exact interventional (marginal) Shapley values by full 2^d coalition enumeration.

    Parameters:
      f: callable mapping (N, d) array to (N,) array of predictions.
      X_explain: (n_explain, d) array of instances to explain.
      X_background: (n_background, d) baseline/reference dataset.
      max_d: maximum allowed feature dimension (default 10).
      chunk_size: number of explain instances evaluated per chunk for memory bounding.

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

    masks, weights, _, _ = _precompute_coalitions(d)
    n_subsets = 1 << d

    # Pre-evaluate v(empty) = mean(f(X_bg))
    bg_preds = f(X_bg)
    v_empty = float(np.mean(bg_preds))

    phi_all = np.zeros((n_explain, d), dtype=np.float64)

    # Process X_explain in chunks
    for c_start in range(0, n_explain, chunk_size):
        c_end = min(c_start + chunk_size, n_explain)
        X_chunk = X_exp[c_start:c_end]
        n_c = len(X_chunk)

        # v_table: shape (n_c, 2^d) stores v(S) for each chunk row and each subset S
        v_table = np.zeros((n_c, n_subsets), dtype=np.float64)
        v_table[:, 0] = v_empty

        # Subset full (index n_subsets - 1): v(D) = f(X_chunk)
        v_table[:, n_subsets - 1] = f(X_chunk)

        # Intermediate subsets (1 to n_subsets - 2)
        for s_idx in range(1, n_subsets - 1):
            mask_s = masks[s_idx]  # boolean array of shape (d,)

            # Construct hybrid input of shape (n_c, n_bg, d)
            # where for col in S: take from X_chunk; else: take from X_bg
            # Efficiently tile and assign:
            # (n_c, 1, d) broadcast with (1, n_bg, d)
            hybrid = np.empty((n_c, n_bg, d), dtype=np.float64)
            hybrid[:, :, mask_s] = X_chunk[:, np.newaxis, mask_s]
            hybrid[:, :, ~mask_s] = X_bg[np.newaxis, :, ~mask_s]

            # Flatten to (n_c * n_bg, d)
            flat_hybrid = hybrid.reshape(-1, d)
            preds = f(flat_hybrid).reshape(n_c, n_bg)
            v_table[:, s_idx] = np.mean(preds, axis=1)

        # Compute Shapley values for each feature j:
        # phi_j = sum_{S subseteq D \ {j}} w(|S|) * (v(S U {j}) - v(S))
        for j in range(d):
            # Mask without feature j: bit j is 0
            # S with feature j: S | (1 << j)
            for s_without in range(n_subsets):
                if not ((s_without >> j) & 1):
                    s_with = s_without | (1 << j)
                    w = weights[s_without]
                    diff = v_table[:, s_with] - v_table[:, s_without]
                    phi_all[c_start:c_end, j] += w * diff

    return phi_all


def mean_abs_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    max_d: int = 10,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute mean absolute Shapley values per feature across instances.

    Returns:
      (d,) array containing (1 / N) * sum_i |phi_j(x_i)|.
    """
    phi = exact_shap(f, X_explain, X_background, max_d=max_d, chunk_size=chunk_size)
    return np.mean(np.abs(phi), axis=0)
