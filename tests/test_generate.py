"""Tests for SCM benchmark generation CLI and artifact outputs."""

import tempfile
import json
from pathlib import Path
import pandas as pd
import pytest

from src.scm.generate import generate_benchmark_from_config
from src.scm.scm import SCM


def test_generate_benchmark_artifacts_and_schema():
    test_config = {
        "n_scms_per_family": 1,
        "d_range": [5, 5],
        "base_seed": 100,
        "nonlinearity_prob": 0.4,
        "noise_type": "gaussian",
        "noise_scale": 1.0,
        "n_samples": 500,
        "standardize": True,
        "clip_bound": 10.0,
        "families": [
            "chain_mediation",
            "confounded_fork",
        ],
        "splits": {
            "train_families": ["chain_mediation"],
            "heldout_families": ["confounded_fork"],
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_df = generate_benchmark_from_config(test_config, out_dir, workers=1, force=True)

        assert len(manifest_df) == 2

        # Required manifest columns
        required_cols = [
            "scm_id",
            "family",
            "d",
            "n_nonlinear",
            "noise_type",
            "noise_scale",
            "has_confounding",
            "mediation_share",
            "standardize",
            "clip_bound",
            "clipped_fraction_Y",
            "seed",
            "split",
        ]
        for col in required_cols:
            assert col in manifest_df.columns

        # Verify each SCM directory and artifact files
        for _, row in manifest_df.iterrows():
            scm_id = row["scm_id"]
            scm_dir = out_dir / scm_id
            assert scm_dir.is_dir()

            data_path = scm_dir / "data.parquet"
            scm_path = scm_dir / "scm.json"
            meta_path = scm_dir / "meta.json"
            gt_path = scm_dir / "ground_truth.json"

            assert data_path.exists()
            assert scm_path.exists()
            assert meta_path.exists()
            assert gt_path.exists()

            # 1. Inspect data.parquet
            df = pd.read_parquet(data_path)
            assert len(df) == 500
            for i in range(1, 6):
                assert f"X{i}" in df.columns
            assert "Y" in df.columns
            assert "row_split" in df.columns

            # Check 60/20/20 split
            splits = df["row_split"].value_counts()
            assert splits["train"] == 300
            assert splits["val"] == 100
            assert splits["test"] == 100

            # 2. Inspect scm.json
            loaded_scm = SCM.from_json(scm_path)
            assert loaded_scm.d == 5

            # 3. Inspect meta.json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["scm_id"] == scm_id
            assert meta["n_samples"] == 500

            # 4. Inspect ground_truth.json
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
            assert len(gt["tau"]) == 5
            assert len(gt["tau_se"]) == 5
            assert len(gt["signed_tau"]) == 5
            assert len(gt["signed_tau_se"]) == 5
            assert len(gt["oracle_mean_abs_shap"]) == 5
            assert len(gt["explain_row_indices"]) == 100
            assert len(gt["background_row_indices"]) == 50
            assert "k_oracle" in gt
            assert "estimand_gap" in gt
            assert "normalized_l1" in gt["estimand_gap"]
            assert "normalized_l2" in gt["estimand_gap"]
            assert "kendall_tau_b" in gt["estimand_gap"]
            assert "top_k_overlap" in gt["estimand_gap"]
            assert "clipped_fraction_Y" in gt

        # 5. Test caching/resumption (force=False should skip)
        cached_df = generate_benchmark_from_config(test_config, out_dir, workers=1, force=False)
        assert len(cached_df) == 2
