"""Regenerate results/figures from the saved stage-5 tables without refitting anything.

Usage:
  python scripts/make_figures.py --scms-dir data/scms --results-dir results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from src.diagnostic import figures as F  # noqa: E402
from src.diagnostic.dataset import load_meta_table, observable_columns  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scms-dir", default="data/scms")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--targets", nargs="+", default=["total_error_share_l2", "total_error_rank"])
    a = p.parse_args()
    res, figs = Path(a.results_dir), Path(a.results_dir) / "figures"
    manifest = Path(a.scms_dir) / "manifest.csv"
    errors = pd.read_csv(res / "errors.csv").merge(pd.read_csv(manifest)[["scm_id", "family", "split"]], on="scm_id")
    scores = pd.read_csv(res / "tables" / "diagnostic_scores.csv")
    preds = pd.read_csv(res / "tables" / "diagnostic_predictions.csv")
    df = load_meta_table(res / "errors.csv", res / "observables.csv", manifest)
    obs_cols = observable_columns(df)
    show = ["meta_gbm", "baseline_constant", "baseline_max_vif", "baseline_test_error",
            "baseline_disagreement_only", "baseline_stability_only"]

    F.fig_error_decomposition(errors, figs, space="share")
    F.fig_error_decomposition(errors, figs, space="raw")
    F.fig_estimand_gap_by_family(errors, figs)
    for proto in ("logfo", "grouped_cv", "designated_split"):
        F.fig_component_predictability(scores, figs, proto)
    for t in a.targets:
        F.fig_total_error_by_explainer(errors, figs, t)
        F.fig_logfo_performance(scores, figs, t, "spearman", show)
        F.fig_logfo_performance(scores, figs, t, "auroc_high_error", show)
        for proto in ("logfo", "designated_split"):
            F.fig_pred_vs_actual(preds[(preds["target"] == t) & (preds["protocol"] == proto)], figs, t, "meta_gbm", proto)
        fi = res / "tables" / f"feature_importance_{t}.csv"
        if fi.exists():
            F.fig_feature_importance(pd.read_csv(fi), figs, t)
        top_obs = df[obs_cols].corrwith(df[t], method="spearman").abs().sort_values(ascending=False).head(6).index.tolist()
        F.fig_observable_scatter(df, figs, t, top_obs)
    print("figures written to", figs)


if __name__ == "__main__":
    main()
