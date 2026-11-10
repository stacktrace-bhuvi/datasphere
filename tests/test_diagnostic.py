"""Tests for src/diagnostic on a synthetic meta-table with a known observable -> error relation."""
import numpy as np
import pandas as pd
import pytest

from src.diagnostic import (
    BASELINE_FEATURES,
    BaselineDiagnostic,
    STRUCTURAL,
    TARGETS,
    design_matrix,
    designated_split,
    grouped_cv,
    logfo,
    make_meta_model,
    meta_model_factory,
    baseline_factory,
    observable_columns,
    paired_bootstrap,
    run_methods,
    score,
    summarize_predictions,
    wilcoxon_paired,
)

FAMILIES = ["chain_mediation", "confounded_fork", "feature_collider", "erdos_renyi_dag", "layered"]


@pytest.fixture(scope="module")
def meta_table():
    """30 SCMs x 2 models x 3 explainers x 2 seeds; error = f(observables) + noise."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        fam = FAMILIES[i % 5]
        base = {"scm_id": f"scm_{i:03d}", "family": fam, "split": "heldout" if fam == "layered" else "train",
                "d": 5 + i % 6, "has_confounding": bool(i % 2), "mediation_share": rng.uniform(),
                "max_vif": rng.uniform(1, 6), "mean_abs_corr": rng.uniform(0, 0.6), "max_abs_corr": rng.uniform(0.2, 0.9),
                "mutual_info_max": rng.uniform(0, 1), "n_train": 1200}
        for model in ("xgboost", "mlp"):
            r2 = rng.uniform(0.3, 0.95)
            for ex in ("kernelshap", "lime", "permutation"):
                for seed in (0, 1):
                    dis = rng.uniform(0, 0.8)
                    stab = rng.uniform(0.5, 1.0)
                    err = 0.4 * dis + 0.3 * (1 - r2) + 0.2 * (1 - stab) + 0.1 * base["mean_abs_corr"] + 0.03 * rng.normal()
                    rows.append({**base, "model": model, "explainer": ex, "seed": seed, "test_r2": r2,
                                 "test_mse": 1 - r2, "disagreement_rank_mean": dis, "disagreement_l2_mean": dis / 2,
                                 "seed_stability_rank": stab, "perturb_stability_rank": stab,
                                 "total_error_rank": err, "total_error_share_l2": err / 2})
    return pd.DataFrame(rows)


def test_observable_columns_exclude_targets_and_structure(meta_table):
    cols = observable_columns(meta_table)
    assert "total_error_rank" not in cols and "mediation_share" not in cols and "has_confounding" not in cols
    assert {"max_vif", "test_r2", "disagreement_rank_mean", "n_train", "d"} <= set(cols)
    X, names = design_matrix(meta_table)
    assert "model_xgboost" in names and "explainer_lime" in names and len(X) == len(meta_table)
    assert not any(c in STRUCTURAL or c in TARGETS for c in names)


def test_grouped_cv_never_leaks_an_scm(meta_table):
    preds = grouped_cv(meta_table, "total_error_rank", meta_model_factory("ridge"), n_splits=5)
    assert preds["pred"].notna().all()
    # every scm_id sits in exactly one fold
    assert (preds.groupby("scm_id")["fold"].nunique() == 1).all()
    assert score(preds["y"], preds["pred"])["spearman"] > 0.8


def test_logfo_holds_out_each_family(meta_table):
    preds = logfo(meta_table, "total_error_rank", meta_model_factory("gbm"))
    assert set(preds["fold"]) == set(FAMILIES)
    for f in FAMILIES:
        g = preds[preds["fold"] == f]
        assert (g["family"] == f).all()
    s = summarize_predictions(preds)
    assert set(s["fold"]) == set(FAMILIES) | {"pooled"}
    assert s.loc[s["fold"] == "pooled", "spearman"].iloc[0] > 0.7


def test_designated_split_uses_manifest_split(meta_table):
    preds = designated_split(meta_table, "total_error_rank", meta_model_factory("rf"))
    assert (preds["family"] == "layered").all()
    assert score(preds["y"], preds["pred"], high_threshold=0.3)["auroc_high_error"] > 0.7


def test_baselines_and_constant(meta_table):
    X, _ = design_matrix(meta_table)
    y = meta_table["total_error_rank"]
    c = BaselineDiagnostic("constant").fit(X, y)
    assert np.allclose(c.predict(X), y.mean())
    for name in BASELINE_FEATURES:
        b = BaselineDiagnostic(name).fit(X, y)
        assert b.predict(X).shape == (len(X),)
    with pytest.raises(ValueError):
        BaselineDiagnostic("nope")


def test_meta_models_fit_and_beat_constant(meta_table):
    methods = {"meta_gbm": meta_model_factory("gbm"), "meta_ridge": meta_model_factory("ridge"),
               "baseline_constant": baseline_factory("constant"), "baseline_max_vif": baseline_factory("max_vif")}
    scores, preds = run_methods(meta_table, "total_error_rank", methods, "logfo")
    pooled = scores[scores["fold"] == "pooled"].set_index("method")
    assert pooled.loc["meta_gbm", "r2"] > pooled.loc["baseline_constant", "r2"]
    assert pooled.loc["meta_ridge", "spearman"] > pooled.loc["baseline_max_vif", "spearman"]
    pa = preds[preds["method"] == "meta_ridge"]
    pb = preds[preds["method"] == "baseline_constant"]
    pbt = paired_bootstrap(pa, pb, n_boot=200)
    assert pbt["mae_gain"] > 0 and pbt["mae_gain_ci_lo"] > 0 and pbt["n_scms"] == 30
    w = wilcoxon_paired(pa, pb)
    assert w["wilcoxon_p"] < 0.05 and w["frac_scms_a_better"] > 0.8


def test_make_meta_model_rejects_unknown():
    with pytest.raises(ValueError):
        make_meta_model("svm")
