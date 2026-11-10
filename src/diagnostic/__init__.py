"""Diagnostic meta-models predicting explanation error from observables (pipeline stage 5)."""

from src.diagnostic.baselines import BASELINE_FEATURES, BaselineDiagnostic, baseline_factory
from src.diagnostic.dataset import KEYS, STRUCTURAL, TARGETS, design_matrix, load_meta_table, observable_columns
from src.diagnostic.evaluation import (
    PROTOCOLS,
    designated_split,
    grouped_cv,
    logfo,
    run_methods,
    score,
    summarize_predictions,
)
from src.diagnostic.meta_regressor import META_MODEL_NAMES, make_meta_model, meta_model_factory
from src.diagnostic.report import feature_importance, run_diagnostic
from src.diagnostic.stats import bootstrap_metric_ci, paired_bootstrap, wilcoxon_paired

__all__ = [
    "BASELINE_FEATURES", "BaselineDiagnostic", "baseline_factory", "KEYS", "STRUCTURAL", "TARGETS",
    "design_matrix", "load_meta_table", "observable_columns", "PROTOCOLS", "designated_split",
    "grouped_cv", "logfo", "run_methods", "score", "summarize_predictions", "META_MODEL_NAMES",
    "make_meta_model", "meta_model_factory", "feature_importance", "run_diagnostic",
    "bootstrap_metric_ci", "paired_bootstrap", "wilcoxon_paired",
]
