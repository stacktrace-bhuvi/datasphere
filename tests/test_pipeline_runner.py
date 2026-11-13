"""Tests for src/metrics/pipeline_runner.py using artifacts written by the real save_attribution()."""

import json

import numpy as np
import pandas as pd
import pytest

from src.explainers import (
    Attribution, attribution_from_reference, compute_reference_shap, explain, save_attribution,
)
from src.metrics.pipeline_runner import ERROR_COLUMNS, PipelineInputError, build_error_table
from src.models import TrainedModel

D = 4
EXPLAIN_IDX, BG_IDX = list(range(100, 112)), list(range(5, 14))


class _Quad:
    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] * X[:, 1] + X[:, 2] ** 2 + 0.3 * X[:, 3]


def _write_scm(scms_dir, scm_id="scm_t", tau=(0.5, 0.4, 0.3, 0.1), oracle=(0.2, 0.2, 0.3, 0.1)):
    d = scms_dir / scm_id
    d.mkdir(parents=True)
    gt = {"scm_id": scm_id, "d": D, "tau": list(tau), "oracle_mean_abs_shap": list(oracle),
          "explain_row_indices": EXPLAIN_IDX, "background_row_indices": BG_IDX}
    (d / "ground_truth.json").write_text(json.dumps(gt))
    return gt


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(1)
    X_exp, X_bg = rng.normal(size=(12, D)), rng.normal(size=(9, D))
    model = TrainedModel("xgboost", 0, _Quad(), [f"X{j}" for j in range(1, D + 1)], {})
    scms, attrs = tmp_path / "scms", tmp_path / "attr"
    _write_scm(scms)
    ref = compute_reference_shap(model, X_exp, X_bg, EXPLAIN_IDX, BG_IDX)
    save_attribution(attrs, "scm_t", "xgboost", 0, attribution_from_reference(ref), EXPLAIN_IDX, reference=ref)
    lime = explain(model, "lime", X_exp, X_bg, seed=0, X_train=X_bg)
    save_attribution(attrs, "scm_t", "xgboost", 0, lime, EXPLAIN_IDX, reference=ref)
    glob = Attribution("permutation", np.tile([0.5, 0.1, 0.9, 0.0], (12, 1)), np.zeros(12), 0.1, {}, True)
    save_attribution(attrs, "scm_t", "xgboost", 0, glob, EXPLAIN_IDX, reference=ref)
    return dict(scms=scms, attrs=attrs, out=tmp_path / "results" / "errors.csv", ref=ref,
                tmp=tmp_path, X_exp=X_exp, X_bg=X_bg, model=model)


def _run(w, **kw):
    return build_error_table(w["scms"], w["attrs"], w["out"], **kw)


def test_produces_documented_columns_and_rows(world):
    df = _run(world)
    assert list(df.columns) == ERROR_COLUMNS
    assert list(pd.read_csv(world["out"]).columns) == ERROR_COLUMNS
    assert sorted(df.explainer) == ["exact", "lime", "permutation"]
    assert df[["estimand_gap", "model_misfit", "total_error_rank"]].notna().all().all()


def test_values_use_the_intended_inputs(world):
    df = _run(world).set_index("explainer")
    tau, oracle = np.array([0.5, 0.4, 0.3, 0.1]), np.array([0.2, 0.2, 0.3, 0.1])
    fitted = world["ref"].mean_abs
    assert df.loc["exact", "explainer_error"] == pytest.approx(0.0, abs=1e-12)     # exact == reference
    assert df.loc["exact", "estimand_gap"] == pytest.approx(np.linalg.norm(tau - oracle) / np.linalg.norm(tau), rel=1e-9)
    assert df.loc["exact", "model_misfit"] == pytest.approx(np.linalg.norm(oracle - fitted) / np.linalg.norm(oracle), rel=1e-9)
    assert df.loc["lime", "explainer_error"] > 1e-6                                  # approximate explainer
    # component terms do not depend on the explainer
    assert df["estimand_gap"].nunique() == 1 and df["model_misfit"].nunique() == 1


def test_global_explainer_has_no_shap_unit_l2_metrics(world):
    row = _run(world).set_index("explainer").loc["permutation"]
    assert np.isnan(row["total_error_l2"]) and np.isnan(row["explainer_error"])
    assert np.isfinite(row["total_error_rank"]) and np.isfinite(row["model_misfit"])


def test_rerun_is_deterministic_and_not_duplicating(world):
    _run(world)
    first = world["out"].read_bytes()
    _run(world)
    assert world["out"].read_bytes() == first
    assert len(pd.read_csv(world["out"])) == 3
    assert not list(world["out"].parent.glob("*.tmp"))


def test_csv_roundtrip_preserves_values(world):
    df = _run(world)
    back = pd.read_csv(world["out"])
    pd.testing.assert_frame_equal(df, back, check_dtype=False)


def test_legacy_sidecar_without_reference_is_rejected_and_nothing_written(world):
    p = world["attrs"] / "scm_t__xgboost__lime__seed0.json"
    side = json.loads(p.read_text())
    del side["fitted_model_mean_abs_shap"], side["reference"]
    p.write_text(json.dumps(side))
    with pytest.raises(PipelineInputError, match="run_attributions"):
        _run(world)
    assert not world["out"].exists()


def test_allow_skipped_keeps_valid_rows_and_reports_the_rest(world):
    p = world["attrs"] / "scm_t__xgboost__lime__seed0.json"
    side = json.loads(p.read_text())
    del side["fitted_model_mean_abs_shap"]
    p.write_text(json.dumps(side))
    df = _run(world, strict=False)
    assert sorted(df.explainer) == ["exact", "permutation"]
    skipped = pd.read_csv(world["out"].parent / "errors_skipped.csv")
    assert skipped["sidecar"].tolist() == ["scm_t__xgboost__lime__seed0.json"]


def _edit(world, name, fn):
    p = world["attrs"] / name
    side = json.loads(p.read_text())
    fn(side)
    p.write_text(json.dumps(side))


@pytest.mark.parametrize("fn,match", [
    (lambda s: s.update(fitted_model_mean_abs_shap=[0.1, 0.2, 0.3]), "length 3"),
    (lambda s: s.update(mean_abs_attributions=[0.1, 0.2, 0.3]), "length 3"),
    (lambda s: s.update(mean_abs_attributions=[0.1, None, 0.3, 0.4]), "numeric|non-finite"),
    (lambda s: s.update(mean_abs_attributions=[0.1, -0.2, 0.3, 0.4]), "non-negative"),
    (lambda s: s["reference"].update(explain_row_indices=list(range(12))), "different explain/background rows"),
    (lambda s: s["reference"].update(feature_names=["X2", "X1", "X3", "X4"]), "feature order"),
    (lambda s: s.update(n_explained_rows=99), "n_explained_rows"),
    (lambda s: s.pop("model"), "missing required"),
    (lambda s: s.update(mean_abs_attributions=[9.0, 9.0, 9.0, 9.0]), "disagrees"),
])
def test_malformed_sidecars_are_rejected(world, fn, match):
    _edit(world, "scm_t__xgboost__lime__seed0.json", fn)
    with pytest.raises(PipelineInputError, match=match):
        _run(world)


def test_filename_content_mismatch_and_duplicates_are_rejected(world):
    src = world["attrs"] / "scm_t__xgboost__lime__seed0.json"
    (world["attrs"] / "scm_t__xgboost__lime__seed1.json").write_text(src.read_text())
    with pytest.raises(PipelineInputError, match="filename does not match"):
        _run(world)


def test_missing_parquet_is_rejected(world):
    (world["attrs"] / "scm_t__xgboost__lime__seed0.parquet").unlink()
    with pytest.raises(PipelineInputError, match="parquet"):
        _run(world)


def test_missing_or_malformed_ground_truth(world):
    gt = world["scms"] / "scm_t" / "ground_truth.json"
    gt.write_text(json.dumps({"scm_id": "scm_t", "tau": [1, 2, 3], "oracle_mean_abs_shap": [1, 2, 3, 4],
                              "explain_row_indices": EXPLAIN_IDX, "background_row_indices": BG_IDX}))
    with pytest.raises(PipelineInputError, match="length 4|expected d=3"):
        _run(world)
    gt.unlink()
    with pytest.raises(PipelineInputError, match="generate"):
        _run(world)


def test_empty_and_missing_directories(tmp_path):
    (tmp_path / "attr").mkdir()
    with pytest.raises(PipelineInputError, match="no attribution sidecars"):
        build_error_table(tmp_path, tmp_path / "attr", tmp_path / "o.csv")
    with pytest.raises(PipelineInputError, match="not found"):
        build_error_table(tmp_path, tmp_path / "nope", tmp_path / "o.csv")
