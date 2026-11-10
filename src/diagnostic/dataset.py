"""Assemble the meta-learning table: observables (inputs) joined to explanation error (targets).

Rows are (scm_id, model, explainer, seed). Structural columns from the manifest
(family, split, has_confounding, mediation_share) are attached for ANALYSIS and for
building held-out splits only; they are never passed to a diagnostic model because a
practitioner cannot observe them. Feature columns are everything in observables.csv
except the keys, plus one-hot indicators of the model and explainer (which ARE known).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["scm_id", "model", "explainer", "seed"]

# Columns of the merged frame that are structural / ground-truth-derived and must never
# be used as diagnostic inputs.
STRUCTURAL = ["family", "split", "has_confounding", "mediation_share", "n_nonlinear",
              "noise_type", "noise_scale", "outcome_noise_scale", "standardize", "clip_bound",
              "clipped_fraction_Y", "attempts", "seed_scm"]

TARGETS = [
    "total_error_rank",        # 1 - Kendall tau-b(tau, attribution)       -- primary
    "total_error_share_l2",    # L2 between importance shares               -- primary magnitude
    "total_error_l2",          # raw normalized L2 (SCHEMAS.md)
    "estimand_gap_share", "model_misfit_share", "explainer_error_share",
    "estimand_gap", "model_misfit", "explainer_error",
    "estimand_gap_rank", "model_misfit_rank", "explainer_error_rank",
    "top_k_overlap",
]


def load_meta_table(errors_csv: str | Path, observables_csv: str | Path,
                    manifest_csv: str | Path) -> pd.DataFrame:
    err = pd.read_csv(errors_csv)
    obs = pd.read_csv(observables_csv)
    man = pd.read_csv(manifest_csv).rename(columns={"seed": "seed_scm"})
    df = err.merge(obs, on=KEYS, how="inner", suffixes=("", "_obs"))
    if "d" in df.columns and "d" in man.columns:
        man = man.drop(columns=["d"])
    df = df.merge(man, on="scm_id", how="left")
    if df.empty:
        raise ValueError("errors.csv and observables.csv share no (scm_id, model, explainer, seed) rows")
    return df.sort_values(KEYS).reset_index(drop=True)


def observable_columns(df: pd.DataFrame) -> list[str]:
    """Numeric observable columns = everything that is neither a key, a target nor structural."""
    drop = set(KEYS) | set(TARGETS) | set(STRUCTURAL)
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def design_matrix(df: pd.DataFrame, features: list[str] | None = None,
                  include_method_indicators: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names). Adds one-hot model/explainer indicators when requested."""
    feats = list(features) if features is not None else observable_columns(df)
    X = df[feats].astype(float).copy()
    if include_method_indicators:
        for name in ("model", "explainer"):
            dummies = pd.get_dummies(df[name], prefix=name, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, list(X.columns)
