"""Tests for src/observables: every quantity must be computable without ground truth."""
import numpy as np
import pandas as pd
import pytest

from src.explainers import explain
from src.models import TrainedModel, split_xy, train_model
from src.observables import (
    concentration_observables,
    cross_model_disagreement,
    data_observables,
    explainer_disagreement,
    gini,
    mutual_information_matrix,
    observables_for_scm,
    perturb_inputs,
    perturbation_observables,
    rank_disagreement,
    seed_stability,
    share_disagreement,
    variance_inflation_factors,
)
from src.scm.scm import SCM


# ------------------------------------------------------------------ dependence
def test_vif_is_one_for_independent_and_large_for_collinear():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    assert np.allclose(variance_inflation_factors(X), 1.0, atol=0.05)
    X[:, 3] = X[:, 0] + X[:, 1] + 0.05 * rng.normal(size=2000)   # near-collinear
    vif = variance_inflation_factors(X)
    assert vif[3] > 50 and vif[2] < 1.1


def test_mutual_information_matrix_detects_dependence():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 3))
    X[:, 1] = np.sin(X[:, 0]) + 0.1 * rng.normal(size=1500)
    M = mutual_information_matrix(X, seed=0)
    assert M.shape == (3, 3) and np.allclose(np.diag(M), 0.0) and np.allclose(M, M.T)
    assert M[0, 1] > 5 * max(M[0, 2], M[1, 2], 1e-3)


def test_data_observables_keys_and_ranges():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=600)
    obs = data_observables(X, y)
    for k in ["max_abs_corr", "mean_abs_corr", "max_vif", "mean_vif", "mutual_info_max", "mutual_info_mean",
              "max_abs_corr_xy", "mi_xy_max", "n_train", "d"]:
        assert k in obs
    assert obs["n_train"] == 600 and obs["d"] == 5
    assert 0 <= obs["max_abs_corr"] <= 1 and obs["max_vif"] >= 1
    assert obs["max_abs_corr_xy"] > 0.5   # X1 drives y


# ------------------------------------------------------------------ concentration / stability
def test_gini_extremes():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert gini([0, 0, 0, 0]) == 0.0


def test_concentration_observables():
    V = np.zeros((50, 4))
    V[:, 0] = 1.0                      # one feature takes all credit, always positive
    c = concentration_observables(V)
    assert c["top1_share"] == pytest.approx(1.0)
    assert c["effective_features"] == pytest.approx(1.0)
    assert c["sign_consistency"] == pytest.approx(1.0)
    V[:, 0] = np.where(np.arange(50) % 2 == 0, 1.0, -1.0)   # sign flips every row
    assert concentration_observables(V)["sign_consistency"] == pytest.approx(0.5)


def test_seed_stability_identical_and_reversed():
    a = np.array([4.0, 3.0, 2.0, 1.0])
    s = seed_stability([a, a, a])
    assert s["seed_stability_rank"] == pytest.approx(1.0) and s["seed_stability_l2"] == pytest.approx(0.0)
    s = seed_stability([a, a[::-1]])
    assert s["seed_stability_rank"] == pytest.approx(-1.0)
    assert np.isnan(seed_stability([a])["seed_stability_rank"])


def test_perturbation_observables_zero_when_unchanged():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(20, 5))
    X = rng.normal(size=(20, 5))
    Xp = perturb_inputs(X, X, sigma=0.05, seed=0)
    assert not np.allclose(X, Xp) and np.allclose(X, Xp, atol=0.5)
    r = perturbation_observables(A, A, X, Xp)
    assert r["perturb_stability_rank"] == pytest.approx(1.0)
    assert r["perturb_rel_change"] == pytest.approx(0.0) and r["sensitivity"] == pytest.approx(0.0)


# ------------------------------------------------------------------ disagreement
def test_disagreement_metrics():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])
    assert rank_disagreement(a, a) == pytest.approx(0.0) and rank_disagreement(a, b) == pytest.approx(2.0)
    assert share_disagreement(a, a) == pytest.approx(0.0) and share_disagreement(a, b) > 0
    out = explainer_disagreement("kernelshap", {"kernelshap": a, "lime": b, "permutation": a}, reference="lime")
    assert out["disagreement_rank_vs_lime"] == pytest.approx(2.0)
    assert out["disagreement_rank_mean"] == pytest.approx(1.0)
    cm = cross_model_disagreement(a, b)
    assert cm["cross_model_disagreement_rank"] == pytest.approx(2.0)


# ------------------------------------------------------------------ end to end on one SCM
@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Generate one SCM and real attributions for it (xgboost, 2 seeds, 3 cheap explainers)."""
    from src.explainers import save_attribution
    from src.scm.ground_truth import compute_ground_truth

    root = tmp_path_factory.mktemp("obs")
    scm = SCM.build("confounded_fork", 5, seed=11, nonlinearity_prob=0.3, outcome_noise_scale=0.4)
    df = scm.sample(500, seed=11)
    df["row_split"] = np.array(["train"] * 300 + ["val"] * 100 + ["test"] * 100)
    scm_dir = root / "scms" / "scm_t"
    scm_dir.mkdir(parents=True)
    df.to_parquet(scm_dir / "data.parquet", index=False)
    gt = compute_ground_truth(scm, df, n_repeats=2, n_ref=10, n_noise_per_ref=200)
    import json
    (scm_dir / "ground_truth.json").write_text(json.dumps(gt))
    pd.DataFrame([{"scm_id": "scm_t", "family": "confounded_fork", "d": 5, "split": "train"}]).to_csv(
        root / "scms" / "manifest.csv", index=False)

    attr_dir = root / "results" / "attributions"
    feats = scm.features
    X_exp = df.loc[gt["explain_row_indices"], feats].to_numpy()
    X_bg = df.loc[gt["background_row_indices"], feats].to_numpy()
    parts = split_xy(df, feats)
    for seed in (0, 1):
        m = train_model("xgboost", df, seed=seed, config={"n_estimators": 80})
        for ex in ("treeshap", "lime", "permutation"):
            a = explain(m, ex, X_exp, X_bg, seed=seed, X_train=parts["train"][0],
                        X_test=parts["test"][0], y_test=parts["test"][1])
            save_attribution(attr_dir, "scm_t", "xgboost", seed, a, gt["explain_row_indices"], m.config)
    return root, scm_dir, attr_dir


def test_observables_for_scm_shape_and_no_ground_truth_columns(tiny_run):
    root, scm_dir, attr_dir = tiny_run
    rows = observables_for_scm(scm_dir, attr_dir, sigma=0.05)
    df = pd.DataFrame(rows)
    assert len(df) == 6                      # 2 seeds x 3 explainers
    assert set(df["explainer"]) == {"treeshap", "lime", "permutation"}
    for forbidden in ("tau", "oracle", "estimand", "family"):
        assert not any(forbidden in c for c in df.columns)
    assert df["seed_stability_rank"].notna().all()        # two seeds available
    assert df["perturb_stability_rank"].between(-1, 1).all()
    assert df["disagreement_rank_vs_lime"].notna().all()
    assert (df["n_train"] == 300).all() and (df["d"] == 5).all()
    assert np.isnan(df["cross_model_disagreement_rank"]).all()   # only one model trained
