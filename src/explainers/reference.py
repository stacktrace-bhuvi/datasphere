"""Reference (exact) SHAP of the FITTED model: Phi(f) in the three-way error decomposition.

    tau  ->  Phi(f*)  ->  Phi(f)  ->  E_hat(f)
             oracle       THIS       selected explainer

`Phi(f)` is computed by brute-force coalition enumeration (`src.scm.oracle.exact_shap`)
on the fitted model, on exactly the explain/background rows used by the ground truth.
It is computed independently of whichever explainer is being evaluated, so it is never a
copy of an empirical attribution and never the true-SCM oracle (`oracle_mean_abs_shap`).

Cost is n_explain * n_background * 2^d model evaluations (5.1M for 100 x 50 x d=10), so
callers compute it ONCE per (scm, model, seed) and attach it to every explainer's sidecar.
Exact enumeration is only supported for d <= 10; larger d raises instead of silently
approximating.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.explainers.attribution import Attribution
from src.models.train import TrainedModel
from src.scm.oracle import exact_shap

REFERENCE_METHOD = "exact_shap_enumeration"
REFERENCE_FIELD = "fitted_model_mean_abs_shap"
REFERENCE_META_FIELD = "reference"
EFFICIENCY_ATOL = 1e-6


class MissingReferenceError(ValueError):
    """An attribution sidecar has no (or an unusable) fitted-model reference SHAP."""


def feature_names_for(d: int) -> list[str]:
    return [f"X{j}" for j in range(1, d + 1)]


@dataclass
class ReferenceShap:
    """Exact SHAP of one fitted model on a fixed explain/background split."""

    values: np.ndarray                 # (n_explain, d) local exact SHAP values
    base_values: np.ndarray            # (n_explain,) E_bg[f(X)]
    feature_names: list[str]
    explain_row_indices: list[int]
    background_row_indices: list[int]
    compute_time_seconds: float
    method: str = REFERENCE_METHOD

    @property
    def mean_abs(self) -> np.ndarray:
        return np.mean(np.abs(self.values), axis=0)

    def sidecar_fields(self, shared_with_exact_explainer: bool = False) -> dict[str, Any]:
        """JSON fields merged into every attribution sidecar of this (scm, model, seed)."""
        meta: dict[str, Any] = {
            "method": self.method,
            "feature_names": list(self.feature_names),
            "n_explain": int(self.values.shape[0]),
            "n_background": int(len(self.background_row_indices)),
            "explain_row_indices": [int(i) for i in self.explain_row_indices],
            "background_row_indices": [int(i) for i in self.background_row_indices],
            "compute_time_seconds": round(float(self.compute_time_seconds), 4),
        }
        if shared_with_exact_explainer:
            meta["shared_with_exact_explainer"] = True
        return {REFERENCE_FIELD: self.mean_abs.tolist(), REFERENCE_META_FIELD: meta}


def compute_reference_shap(model: TrainedModel, X_explain: np.ndarray, X_background: np.ndarray,
                           explain_row_indices: Sequence[int],
                           background_row_indices: Sequence[int]) -> ReferenceShap:
    """Exact interventional SHAP of `model` on the ground-truth explain/background rows."""
    X_exp = np.asarray(X_explain, dtype=np.float64)
    X_bg = np.asarray(X_background, dtype=np.float64)
    if X_exp.ndim != 2 or X_bg.ndim != 2 or X_exp.shape[1] != X_bg.shape[1]:
        raise ValueError(f"X_explain {X_exp.shape} and X_background {X_bg.shape} must be 2-D "
                         "with the same number of features")
    d = X_exp.shape[1]
    if len(model.feature_names) != d:
        raise ValueError(f"model has {len(model.feature_names)} features but data has {d}")
    if len(explain_row_indices) != len(X_exp) or len(background_row_indices) != len(X_bg):
        raise ValueError("row-index lists must match the number of explain/background rows")
    if not (np.all(np.isfinite(X_exp)) and np.all(np.isfinite(X_bg))):
        raise ValueError("explain/background rows must be finite")

    t0 = time.perf_counter()
    phi = exact_shap(model.predict, X_exp, X_bg)   # raises ValueError for d > 10
    elapsed = time.perf_counter() - t0
    base = np.full(len(X_exp), float(model.predict(X_bg).mean()))

    if phi.shape != X_exp.shape or not np.all(np.isfinite(phi)):
        raise ArithmeticError("reference SHAP has wrong shape or non-finite values")
    # Efficiency axiom: sum_j phi_j(x) = f(x) - E_bg[f]. Catches nondeterministic predict().
    gap = np.abs(phi.sum(axis=1) - (model.predict(X_exp) - base))
    if gap.max() > EFFICIENCY_ATOL:
        raise ArithmeticError(f"reference SHAP violates efficiency (max gap {gap.max():.3e})")

    return ReferenceShap(phi, base, list(model.feature_names), [int(i) for i in explain_row_indices],
                         [int(i) for i in background_row_indices], elapsed)


def attribution_from_reference(ref: ReferenceShap, config: dict[str, Any] | None = None) -> Attribution:
    """The `exact` explainer IS the reference algorithm; reuse it instead of recomputing 2^d work."""
    return Attribution("exact", ref.values.copy(), ref.base_values.copy(),
                       ref.compute_time_seconds, dict(config or {}), False)


def read_reference(sidecar: dict[str, Any], source: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (fitted_model_mean_abs_shap, reference_meta) from a sidecar, or raise
    MissingReferenceError with instructions. Never fabricates a value."""
    if REFERENCE_FIELD not in sidecar or REFERENCE_META_FIELD not in sidecar:
        raise MissingReferenceError(
            f"{source}: no '{REFERENCE_FIELD}'/'{REFERENCE_META_FIELD}' (artifact written before the "
            "reference-SHAP integration). Re-run `python scripts/run_attributions.py ...` without "
            "--force to back-fill it into existing sidecars, or with --force to recompute everything.")
    vec = np.asarray(sidecar[REFERENCE_FIELD], dtype=float)
    meta = sidecar[REFERENCE_META_FIELD]
    if vec.ndim != 1 or vec.size == 0 or not np.all(np.isfinite(vec)) or np.any(vec < 0):
        raise MissingReferenceError(f"{source}: '{REFERENCE_FIELD}' must be a finite, non-negative 1-D vector")
    if not isinstance(meta, dict) or meta.get("method") != REFERENCE_METHOD:
        raise MissingReferenceError(f"{source}: reference method must be '{REFERENCE_METHOD}', "
                                    f"got {meta.get('method') if isinstance(meta, dict) else meta!r}")
    return vec, meta
