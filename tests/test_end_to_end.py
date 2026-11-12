"""Small genuine end-to-end run: SCM -> ground truth -> train -> explain (+reference) -> errors -> tables."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.pipeline_runner import PipelineInputError, build_error_table
from src.metrics.results_tables import summarize_errors
from src.scm.ground_truth import compute_ground_truth
from src.scm.scm import SCM

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_attributions", ROOT / "scripts" / "run_attributions.py")
run_attr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_attr)

EXPLAINERS = ["exact", "treeshap", "lime", "permutation"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    scm = SCM.build("chain_mediation", 5, seed=7, nonlinearity_prob=0.4)
    scm.metadata["scm_id"] = "scm_e2e"
    df = scm.sample(500, seed=7)
    df["row_split"] = ["train"] * 300 + ["val"] * 100 + ["test"] * 100
    d = root / "scms" / "scm_e2e"
    d.mkdir(parents=True)
    df.to_parquet(d / "data.parquet")
    gt = compute_ground_truth(scm, df, n_repeats=2, n_ref=5, n_noise_per_ref=200)
    gt["scm_id"] = "scm_e2e"
    (d / "ground_truth.json").write_text(json.dumps(gt))
    attrs = root / "results" / "attributions"
    rows = run_attr.run_scm(d, attrs, ["xgboost"], EXPLAINERS, [0])
    pd.DataFrame(rows).to_csv(root / "results" / "models.csv", index=False)
    return dict(root=root, d=d, attrs=attrs, gt=gt, rows=rows)


def test_artifacts_and_reference_are_written_for_every_explainer(world):
    for ex in EXPLAINERS:
        side = json.loads((world["attrs"] / f"scm_e2e__xgboost__{ex}__seed0.json").read_text())
        assert len(side["fitted_model_mean_abs_shap"]) == 5
        assert side["reference"]["explain_row_indices"] == world["gt"]["explain_row_indices"]
    # one reference per fitted model, shared: identical across explainers, equal to exact's own output
    vecs = [json.loads((world["attrs"] / f"scm_e2e__xgboost__{ex}__seed0.json").read_text())["fitted_model_mean_abs_shap"]
            for ex in EXPLAINERS]
    assert all(v == vecs[0] for v in vecs)
    exact = json.loads((world["attrs"] / "scm_e2e__xgboost__exact__seed0.json").read_text())
    assert exact["mean_abs_attributions"] == exact["fitted_model_mean_abs_shap"]


def test_treeshap_matches_the_independent_reference(world):
    t = json.loads((world["attrs"] / "scm_e2e__xgboost__treeshap__seed0.json").read_text())
    assert np.allclose(t["mean_abs_attributions"], t["fitted_model_mean_abs_shap"], atol=1e-4)


def test_reference_differs_from_true_scm_oracle(world):
    ref = np.array(json.loads((world["attrs"] / "scm_e2e__xgboost__lime__seed0.json").read_text())["fitted_model_mean_abs_shap"])
    assert not np.allclose(ref, world["gt"]["oracle_mean_abs_shap"], atol=1e-6)


def test_errors_and_tables_from_real_artifacts(world, tmp_path):
    df = build_error_table(world["root"] / "scms", world["attrs"], tmp_path / "errors.csv")
    assert sorted(df.explainer) == sorted(EXPLAINERS) and len(df) == 4
    row = df.set_index("explainer")
    assert row.loc["exact", "explainer_error"] == pytest.approx(0.0, abs=1e-9)
    assert row.loc["treeshap", "explainer_error"] < 1e-3
    assert row.loc["lime", "explainer_error"] > row.loc["treeshap", "explainer_error"]
    assert row.loc["exact", "estimand_gap"] == pytest.approx(world["gt"]["estimand_gap"]["normalized_l2"], rel=1e-9)
    assert np.isnan(row.loc["permutation", "total_error_l2"])
    tables = summarize_errors(tmp_path / "errors.csv", tmp_path / "tables", scms_dir=None)
    assert set(tables["by_explainer"]["explainer"]) == set(EXPLAINERS)


def test_rerun_skips_existing_valid_artifacts(world):
    before = {p.name: p.stat().st_mtime_ns for p in world["attrs"].iterdir()}
    run_attr.run_scm(world["d"], world["attrs"], ["xgboost"], EXPLAINERS, [0])
    assert before == {p.name: p.stat().st_mtime_ns for p in world["attrs"].iterdir()}


def _strip_reference(path):
    s = json.loads(path.read_text())
    s.pop("fitted_model_mean_abs_shap"), s.pop("reference")
    path.write_text(json.dumps(s))
    return s


def test_legacy_sidecars_are_rejected_then_backfilled_without_touching_attributions(world, tmp_path):
    attrs = tmp_path / "attr"
    attrs.mkdir()
    for p in world["attrs"].iterdir():
        (attrs / p.name).write_bytes(p.read_bytes())
    legacy = {ex: _strip_reference(attrs / f"scm_e2e__xgboost__{ex}__seed0.json") for ex in EXPLAINERS}
    with pytest.raises(PipelineInputError, match="run_attributions"):
        build_error_table(world["root"] / "scms", attrs, tmp_path / "e.csv")
    parquet_bytes = (attrs / "scm_e2e__xgboost__lime__seed0.parquet").read_bytes()
    run_attr.run_scm(world["d"], attrs, ["xgboost"], EXPLAINERS, [0])
    for ex in EXPLAINERS:
        new = json.loads((attrs / f"scm_e2e__xgboost__{ex}__seed0.json").read_text())
        assert new["mean_abs_attributions"] == legacy[ex]["mean_abs_attributions"]
        assert "fitted_model_mean_abs_shap" in new
    assert (attrs / "scm_e2e__xgboost__lime__seed0.parquet").read_bytes() == parquet_bytes
    assert len(build_error_table(world["root"] / "scms", attrs, tmp_path / "e.csv")) == 4


def test_backfill_refuses_when_retrained_model_differs_from_old_artifacts(world, tmp_path):
    attrs = tmp_path / "attr"
    attrs.mkdir()
    for p in world["attrs"].iterdir():
        (attrs / p.name).write_bytes(p.read_bytes())
    _strip_reference(attrs / "scm_e2e__xgboost__lime__seed0.json")
    exact = attrs / "scm_e2e__xgboost__exact__seed0.json"
    s = json.loads(exact.read_text())
    s["mean_abs_attributions"] = [v + 0.5 for v in s["mean_abs_attributions"]]   # "different model"
    exact.write_text(json.dumps(s))
    with pytest.raises(RuntimeError, match="--force"):
        run_attr.run_scm(world["d"], attrs, ["xgboost"], ["lime"], [0])
