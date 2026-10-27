"""Validation tests ensuring all generated files strictly conform to docs/SCHEMAS.md."""

import json
from pathlib import Path
import pandas as pd
import pytest

from scripts.make_mock_results import make_mock_results
from src.scm.generate import generate_benchmark_from_config
from src.scm.scm import SCM


@pytest.fixture(scope="module")
def populated_dataset_and_results(tmp_path_factory):
    """Fixture that generates a small benchmark and mock results into temporary directories."""
    base_dir = tmp_path_factory.mktemp("schema_test")
    scms_dir = base_dir / "data" / "scms_small"
    results_dir = base_dir / "results"

    cfg = {
        "dataset_name": "scm_benchmark_small",
        "n_scms_per_family": 1,
        "d_range": [5, 6],
        "n_samples": 500,
        "base_seed": 42,
        "nonlinearity_prob": 0.5,
        "interaction_prob": 0.2,
        "noise_type": "gaussian",
        "noise_scale": 1.0,
        "standardize": True,
        "clip_bound": 10.0,
        "families": ["chain_mediation", "confounded_fork"],
        "splits": {
            "train_families": ["chain_mediation"],
            "heldout_families": ["confounded_fork"],
        },
    }
    # 1. Generate small SCM benchmark
    generate_benchmark_from_config(cfg, scms_dir, workers=1, force=True)

    # 2. Generate mock results
    make_mock_results(scms_dir, results_dir)

    return {"scms_dir": scms_dir, "results_dir": results_dir}


def test_validate_data_parquet_schema(populated_dataset_and_results):
    scms_dir = populated_dataset_and_results["scms_dir"]
    manifest_df = pd.read_csv(scms_dir / "manifest.csv")

    for _, row in manifest_df.iterrows():
        scm_id = row["scm_id"]
        d = int(row["d"])
        parquet_path = scms_dir / scm_id / "data.parquet"

        assert parquet_path.exists(), f"Missing data.parquet for {scm_id}"
        df = pd.read_parquet(parquet_path)

        # Check rows and columns
        assert len(df) == 500
        expected_cols = [f"X{i}" for i in range(1, d + 1)] + ["Y", "row_split"]
        assert list(df.columns) == expected_cols

        # Check types
        for f in [f"X{i}" for i in range(1, d + 1)] + ["Y"]:
            assert pd.api.types.is_float_dtype(df[f]), f"Column {f} should be float"

        # Check 60/20/20 seeded split
        splits = df["row_split"].value_counts().to_dict()
        assert splits["train"] == 300
        assert splits["val"] == 100
        assert splits["test"] == 100


def test_validate_scm_and_meta_json(populated_dataset_and_results):
    scms_dir = populated_dataset_and_results["scms_dir"]
    manifest_df = pd.read_csv(scms_dir / "manifest.csv")

    for _, row in manifest_df.iterrows():
        scm_id = row["scm_id"]
        d = int(row["d"])
        scm_dir = scms_dir / scm_id

        # scm.json validation
        scm_json_path = scm_dir / "scm.json"
        assert scm_json_path.exists()
        loaded_scm = SCM.from_json(scm_json_path)
        assert loaded_scm.d == d
        assert "Y" in loaded_scm.graph.nodes

        # meta.json validation
        meta_json_path = scm_dir / "meta.json"
        assert meta_json_path.exists()
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_meta_keys = [
            "scm_id", "family", "d", "n_nonlinear", "noise_type",
            "noise_scale", "has_confounding", "mediation_share",
            "standardize", "clip_bound", "seed", "split", "n_samples",
        ]
        for k in required_meta_keys:
            assert k in meta, f"Missing key {k} in meta.json for {scm_id}"
        assert meta["scm_id"] == scm_id
        assert meta["d"] == d


def test_validate_ground_truth_json(populated_dataset_and_results):
    scms_dir = populated_dataset_and_results["scms_dir"]
    manifest_df = pd.read_csv(scms_dir / "manifest.csv")

    for _, row in manifest_df.iterrows():
        scm_id = row["scm_id"]
        d = int(row["d"])
        gt_path = scms_dir / scm_id / "ground_truth.json"

        assert gt_path.exists()
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        # Structure checks
        assert gt["scm_id"] == scm_id
        assert gt["d"] == d
        assert gt["n_repeats"] == 5

        # Vector dimensions
        assert len(gt["tau"]) == d
        assert len(gt["tau_se"]) == d
        assert len(gt["signed_tau"]) == d
        assert len(gt["signed_tau_se"]) == d
        assert len(gt["oracle_mean_abs_shap"]) == d

        # Fixed indices
        assert len(gt["explain_row_indices"]) == 100
        assert len(gt["background_row_indices"]) == 50

        # Estimand gap metrics
        gap = gt["estimand_gap"]
        for key in ["normalized_l1", "normalized_l2", "kendall_tau_b", "top_k_overlap", "k"]:
            assert key in gap
        assert isinstance(gap["normalized_l1"], float)
        assert isinstance(gap["normalized_l2"], float)
        assert -1.0 <= gap["kendall_tau_b"] <= 1.0
        assert 0.0 <= gap["top_k_overlap"] <= 1.0
        assert gap["k"] == gt["k_oracle"]

        # Clipping info
        assert "clipped_fractions" in gt
        assert "clipped_fraction_Y" in gt
        assert 0.0 <= gt["clipped_fraction_Y"] <= 1.0


def test_validate_attributions_parquet_and_sidecar(populated_dataset_and_results):
    results_dir = populated_dataset_and_results["results_dir"]
    attr_dir = results_dir / "attributions"
    assert attr_dir.exists()

    parquet_files = list(attr_dir.glob("*.parquet"))
    assert len(parquet_files) > 0, "No attribution parquet files found"

    for p_file in parquet_files:
        sidecar_file = p_file.with_suffix(".json")
        assert sidecar_file.exists(), f"Missing JSON sidecar for {p_file}"

        # 1. Parquet verification
        df = pd.read_parquet(p_file)
        assert "row_index" in df.columns
        assert "base_value" in df.columns
        assert len(df) == 100
        assert pd.api.types.is_integer_dtype(df["row_index"])

        # Check features
        feat_cols = [c for c in df.columns if c.startswith("X")]
        assert len(feat_cols) in (5, 6)
        for fc in feat_cols:
            assert pd.api.types.is_float_dtype(df[fc])

        # 2. Sidecar verification
        with open(sidecar_file, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        for key in ["scm_id", "model", "explainer", "seed", "n_explained_rows", "mean_abs_attributions", "compute_time_seconds", "config"]:
            assert key in sidecar, f"Missing key {key} in {sidecar_file}"
        assert sidecar["n_explained_rows"] == 100
        assert len(sidecar["mean_abs_attributions"]) == len(feat_cols)


def test_validate_errors_and_observables_csv(populated_dataset_and_results):
    results_dir = populated_dataset_and_results["results_dir"]

    # 1. errors.csv validation
    errors_csv = results_dir / "errors.csv"
    assert errors_csv.exists()
    errors_df = pd.read_csv(errors_csv)

    expected_error_cols = [
        "scm_id", "model", "explainer", "seed",
        "total_error_l2", "total_error_rank",
        "estimand_gap", "model_misfit", "explainer_error",
    ]
    assert list(errors_df.columns) == expected_error_cols
    assert len(errors_df) > 0

    for col in ["total_error_l2", "total_error_rank", "estimand_gap", "model_misfit", "explainer_error"]:
        assert pd.api.types.is_float_dtype(errors_df[col])

    # 2. observables.csv validation
    obs_csv = results_dir / "observables.csv"
    assert obs_csv.exists()
    obs_df = pd.read_csv(obs_csv)

    expected_obs_cols = [
        "scm_id", "model", "explainer", "seed",
        "max_vif", "mean_vif", "mutual_info_max", "mutual_info_mean",
        "test_mse", "test_r2", "explainer_stability", "explainer_sensitivity",
        "prediction_concentration", "explainer_disagreement_rank",
        "explainer_disagreement_l2", "n_train",
    ]
    assert list(obs_df.columns) == expected_obs_cols
    assert len(obs_df) > 0
    assert (obs_df["n_train"] == 1200).all()
