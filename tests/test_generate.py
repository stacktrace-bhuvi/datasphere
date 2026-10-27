"""Tests for SCM benchmark generation CLI and manifest output."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.scm.generate import generate_benchmark_from_config
from src.scm.scm import SCM


def test_generate_benchmark_and_manifest():
    test_config = {
        "n_scms_per_family": 2,
        "d_range": [5, 7],
        "base_seed": 100,
        "nonlinearity_prob": 0.4,
        "noise_type": "gaussian",
        "noise_scale": 1.0,
        "families": [
            "chain_mediation",
            "confounded_fork",
            "layered",
        ],
        "splits": {
            "train_families": ["chain_mediation", "confounded_fork"],
            "heldout_families": ["layered"],
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_df = generate_benchmark_from_config(test_config, tmpdir)

        # 3 families * 2 = 6 SCMs
        assert len(manifest_df) == 6

        # Required columns
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
            "seed",
            "split",
        ]
        for col in required_cols:
            assert col in manifest_df.columns

        # Verify splits
        layered_rows = manifest_df[manifest_df["family"] == "layered"]
        assert (layered_rows["split"] == "heldout").all()

        train_rows = manifest_df[manifest_df["family"].isin(["chain_mediation", "confounded_fork"])]
        assert (train_rows["split"] == "train").all()

        # Check JSON files exist and can be loaded
        scms_dir = Path(tmpdir) / "scms"
        for _, row in manifest_df.iterrows():
            json_file = scms_dir / f"{row['scm_id']}.json"
            assert json_file.exists()
            loaded_scm = SCM.from_json(json_file)
            assert loaded_scm.d == row["d"]
