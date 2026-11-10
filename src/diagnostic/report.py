"""Stage 5 orchestrator: run every diagnostic under every protocol and write tables + figures.

Outputs (under --out-dir, default results/):
  tables/diagnostic_scores.csv        target x protocol x method x fold -> r2, spearman, mae, rmse, auroc
  tables/diagnostic_predictions.csv   out-of-sample prediction for every row, every method/protocol
  tables/paired_tests.csv             meta-model vs each baseline: paired bootstrap + Wilcoxon over SCMs
  tables/metric_bootstrap_ci.csv      bootstrap CI of pooled Spearman / MAE per method and protocol
  tables/feature_importance_<t>.csv   permutation importance of observables in the meta-model
  tables/error_summary_*.csv          descriptive error tables (results_tables.summarize_errors)
  tables/estimand_gap_by_family.csv   per-family estimand gap statistics
  figures/*.png|pdf
  diagnostic_summary.json             headline numbers used in the report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.diagnostic import figures as F
from src.diagnostic.baselines import BASELINE_FEATURES, baseline_factory
from src.diagnostic.dataset import design_matrix, load_meta_table, observable_columns
from src.diagnostic.evaluation import run_methods
from src.diagnostic.meta_regressor import META_MODEL_NAMES, meta_model_factory
from src.diagnostic.stats import bootstrap_metric_ci, paired_bootstrap, wilcoxon_paired
from src.metrics.results_tables import summarize_errors

DEFAULT_TARGETS = ["total_error_rank", "total_error_share_l2"]
COMPONENT_TARGETS = ["estimand_gap_share", "model_misfit_share", "explainer_error_share"]


def _methods(meta_models, baselines, seed):
    m = {f"meta_{n}": meta_model_factory(n, seed) for n in meta_models}
    m.update({f"baseline_{n}": baseline_factory(n) for n in baselines})
    return m


def feature_importance(df: pd.DataFrame, target: str, meta_model: str = "gbm", seed: int = 0,
                       n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance of each observable: fit on train families, score on held-out family."""
    X, names = design_matrix(df)
    y = df[target].to_numpy(float)
    tr = df["split"].to_numpy() == "train"
    te = ~tr
    model = meta_model_factory(meta_model, seed)()
    model.fit(X[tr], y[tr])
    r = permutation_importance(model, X[te], y[te], n_repeats=n_repeats, random_state=seed, scoring="r2")
    return pd.DataFrame({"feature": names, "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std}).sort_values("importance_mean", ascending=False)


def run_diagnostic(errors_csv: str | Path, observables_csv: str | Path, manifest_csv: str | Path,
                   out_dir: str | Path = "results", targets: list[str] | None = None,
                   component_targets: list[str] | None = None,
                   meta_models: tuple[str, ...] = META_MODEL_NAMES,
                   baselines: tuple[str, ...] = tuple(BASELINE_FEATURES),
                   protocols: tuple[str, ...] = ("grouped_cv", "logfo", "designated_split"),
                   n_boot: int = 2000, seed: int = 0, make_figures: bool = True,
                   verbose: bool = True) -> dict:
    out = Path(out_dir)
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    targets = list(targets or DEFAULT_TARGETS)
    component_targets = list(COMPONENT_TARGETS if component_targets is None else component_targets)

    df = load_meta_table(errors_csv, observables_csv, manifest_csv)
    obs_cols = observable_columns(df)
    if verbose:
        print(f"meta table: {len(df)} rows, {df['scm_id'].nunique()} SCMs, {len(obs_cols)} observables")

    methods = _methods(meta_models, baselines, seed)
    all_scores, all_preds = [], []
    for target in targets + component_targets:
        for protocol in protocols:
            if verbose:
                print(f"  {target:24s} {protocol}", flush=True)
            s, p = run_methods(df, target, methods, protocol)
            all_scores.append(s)
            all_preds.append(p.assign(target=target, protocol=protocol))
    scores = pd.concat(all_scores, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    scores.to_csv(tables / "diagnostic_scores.csv", index=False)
    preds.to_csv(tables / "diagnostic_predictions.csv", index=False)

    # Paired comparisons: best meta-model vs every baseline (and vs other meta-models).
    main_meta = f"meta_{meta_models[0]}"
    tests = []
    cis = []
    for target in targets:
        for protocol in [p for p in protocols if p != "grouped_cv"] + ["grouped_cv"]:
            sub = preds[(preds["target"] == target) & (preds["protocol"] == protocol)]
            pa = sub[sub["method"] == main_meta]
            for m in methods:
                if m == main_meta:
                    continue
                pb = sub[sub["method"] == m]
                row = {"target": target, "protocol": protocol, "method_a": main_meta, "method_b": m}
                row.update(paired_bootstrap(pa, pb, n_boot=n_boot, seed=seed))
                row.update(wilcoxon_paired(pa, pb))
                tests.append(row)
            for m in methods:
                pm = sub[sub["method"] == m]
                ci = {"target": target, "protocol": protocol, "method": m}
                ci.update(bootstrap_metric_ci(pm, "spearman", n_boot=min(n_boot, 500), seed=seed))
                ci.update(bootstrap_metric_ci(pm, "mae", n_boot=min(n_boot, 500), seed=seed))
                cis.append(ci)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(tables / "paired_tests.csv", index=False)
    cis_df = pd.DataFrame(cis)
    cis_df.to_csv(tables / "metric_bootstrap_ci.csv", index=False)

    # Feature importance of the main meta-model on the designated held-out family.
    importances = {}
    if "designated_split" in protocols and (df["split"] == "heldout").any():
        for target in targets:
            imp = feature_importance(df, target, meta_models[0], seed)
            imp.to_csv(tables / f"feature_importance_{target}.csv", index=False)
            importances[target] = imp

    # Descriptive error tables.
    errors = pd.read_csv(errors_csv).merge(pd.read_csv(manifest_csv)[["scm_id", "family", "split"]], on="scm_id")
    summarize_errors(errors_csv, tables)
    gap = errors.groupby(["scm_id", "family"], as_index=False)[["estimand_gap", "estimand_gap_share", "estimand_gap_rank"]].first()
    gap_tab = gap.groupby("family")[["estimand_gap", "estimand_gap_share", "estimand_gap_rank"]].agg(["mean", "std", "count"])
    gap_tab.columns = [f"{a}_{b}" for a, b in gap_tab.columns]
    gap_tab.reset_index().to_csv(tables / "estimand_gap_by_family.csv", index=False)
    by_fam_exp = errors.groupby(["family", "explainer"])[["total_error_rank", "total_error_share_l2"]].mean().reset_index()
    by_fam_exp.to_csv(tables / "error_by_family_explainer.csv", index=False)

    # Figures.
    if make_figures:
        F.fig_error_decomposition(errors, figs, space="share")
        F.fig_error_decomposition(errors, figs, space="raw")
        F.fig_estimand_gap_by_family(errors, figs)
        for t in targets:
            F.fig_total_error_by_explainer(errors, figs, t)
            show = [main_meta] + [m for m in methods if m.startswith("baseline_") and m in
                                 ("baseline_constant", "baseline_max_vif", "baseline_test_error",
                                  "baseline_disagreement_only", "baseline_stability_only")]
            F.fig_logfo_performance(scores, figs, t, "spearman", show)
            F.fig_logfo_performance(scores, figs, t, "auroc_high_error", show)
            for proto in [p for p in protocols if p in ("logfo", "designated_split")]:
                F.fig_pred_vs_actual(preds[(preds["target"] == t) & (preds["protocol"] == proto)], figs, t, main_meta, proto)
            if t in importances:
                F.fig_feature_importance(importances[t], figs, t)
            top_obs = (df[obs_cols].corrwith(df[t], method="spearman").abs().sort_values(ascending=False).head(6).index.tolist())
            F.fig_observable_scatter(df, figs, t, top_obs)

    summary = _headline(scores, tests_df, df, targets, main_meta, obs_cols)
    with open(out / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    if verbose:
        print(json.dumps(summary["headline"], indent=2, default=float))
    return {"scores": scores, "predictions": preds, "paired_tests": tests_df, "metric_ci": cis_df,
            "feature_importance": importances, "meta_table": df, "summary": summary}


def _headline(scores, tests, df, targets, main_meta, obs_cols) -> dict:
    out = {"n_rows": int(len(df)), "n_scms": int(df["scm_id"].nunique()), "n_observables": len(obs_cols),
           "observables": obs_cols, "families": sorted(df["family"].unique().tolist()),
           "models": sorted(df["model"].unique().tolist()), "explainers": sorted(df["explainer"].unique().tolist()),
           "headline": {}}
    for t in targets:
        h = {}
        for proto in scores["protocol"].unique():
            s = scores[(scores["target"] == t) & (scores["protocol"] == proto) & (scores["fold"] == "pooled")]
            s = s.set_index("method")
            best_base = s[s.index.str.startswith("baseline_")]["spearman"].idxmax()
            h[proto] = {
                "meta_spearman": float(s.loc[main_meta, "spearman"]),
                "meta_r2": float(s.loc[main_meta, "r2"]),
                "meta_auroc_high_error": float(s.loc[main_meta, "auroc_high_error"]),
                "best_baseline": best_base,
                "best_baseline_spearman": float(s.loc[best_base, "spearman"]),
                "best_baseline_r2": float(s.loc[best_base, "r2"]),
            }
            rest = scores[(scores["target"] == t) & (scores["protocol"] == proto) & (scores["fold"] != "pooled")
                          & (scores["method"] == main_meta)]
            fam = rest[~rest["fold"].astype(str).str.startswith("by_method")]
            bym = rest[rest["fold"].astype(str).str.startswith("by_method")]
            if len(fam):
                h[proto]["meta_spearman_by_fold"] = dict(zip(fam["fold"], fam["spearman"].round(3)))
            if len(bym):
                h[proto]["meta_spearman_by_method"] = dict(zip(bym["fold"].str.replace("by_method:", ""),
                                                               bym["spearman"].round(3)))
                bb = scores[(scores["target"] == t) & (scores["protocol"] == proto) & (scores["method"] == best_base)
                            & scores["fold"].astype(str).str.startswith("by_method")]
                h[proto]["best_baseline_spearman_by_method"] = dict(zip(bb["fold"].str.replace("by_method:", ""),
                                                                        bb["spearman"].round(3)))
            tt = tests[(tests["target"] == t) & (tests["protocol"] == proto) & (tests["method_b"] == best_base)]
            if len(tt):
                r = tt.iloc[0]
                h[proto]["vs_best_baseline"] = {k: float(r[k]) for k in
                                                ["mae_gain", "mae_gain_ci_lo", "mae_gain_ci_hi", "mae_gain_p",
                                                 "spearman_gain", "spearman_gain_ci_lo", "spearman_gain_ci_hi",
                                                 "spearman_gain_p", "wilcoxon_p", "frac_scms_a_better"]}
        out["headline"][t] = h
    return out
