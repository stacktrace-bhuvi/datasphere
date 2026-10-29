"""Generate synthetic mock results (attributions, errors, observables) conforming to SCHEMAS.md."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import yaml

from src.scm.generate import generate_benchmark_from_config


MODELS = ["xgboost", "mlp"]
EXPLAINERS = ["treeshap", "kernelshap", "lime", "permutation"]
SEEDS = [0, 1]


def make_mock_results(scms_dir: str | Path, results_dir: str | Path) -> None:
    """Generate mock results matching SCHEMAS.md for all SCMs in scms_dir."""
    scms_path = Path(scms_dir)
    res_path = Path(results_dir)
    attr_dir = res_path / "attributions"
    attr_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = scms_path / "manifest.csv"
    if not manifest_csv.exists():
        print(f"Manifest not found in {scms_path}, generating small benchmark suite...")
        small_cfg = Path("configs/scm_small.yaml")
        with open(small_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        manifest_df = generate_benchmark_from_config(cfg, scms_path, workers=2, force=False)
    else:
        manifest_df = pd.read_csv(manifest_csv)

    error_rows: list[dict[str, object]] = []
    observable_rows: list[dict[str, object]] = []

    print(f"Generating mock attributions and metrics for {len(manifest_df)} SCMs...")

    for _, row in manifest_df.iterrows():
        scm_id = str(row["scm_id"])
        d = int(row["d"])
        scm_dir = scms_path / scm_id

        gt_file = scm_dir / "ground_truth.json"
        if not gt_file.exists():
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt = json.load(f)

        explain_indices = gt.get("explain_row_indices", list(range(100)))
        n_explain = len(explain_indices)
        tau = np.array(gt["tau"], dtype=np.float64)
        tau_l2 = float(np.linalg.norm(tau))
        estimand_gap_val = float(gt["estimand_gap"]["normalized_l2"])

        for model in MODELS:
            for explainer in EXPLAINERS:
                # Treeshap only valid for tree models; skip for mlp in mock
                if model == "mlp" and explainer == "treeshap":
                    continue

                for seed in SEEDS:
                    rng = np.random.default_rng(abs(hash((scm_id, model, explainer, seed))) % (2**31))

                    # 1. Generate fake local attributions for explained rows
                    # Shape (n_explain, d) centered around tau with some noise
                    noise_mag = 0.2 if model == "xgboost" else 0.3
                    mock_attrs = np.tile(tau, (n_explain, 1)) + rng.normal(0, noise_mag, size=(n_explain, d))
                    base_values = rng.normal(0.5, 0.1, size=n_explain)

                    attr_df = pd.DataFrame(
                        mock_attrs,
                        columns=[f"X{i}" for i in range(1, d + 1)],
                    )
                    attr_df.insert(0, "row_index", explain_indices)
                    attr_df["base_value"] = base_values

                    # Write Parquet attribution file
                    prefix = f"{scm_id}__{model}__{explainer}__seed{seed}"
                    parquet_file = attr_dir / f"{prefix}.parquet"
                    attr_df.to_parquet(parquet_file, index=False)

                    # Write JSON sidecar
                    sidecar_file = attr_dir / f"{prefix}.json"
                    mean_abs = np.mean(np.abs(mock_attrs), axis=0).tolist()
                    sidecar_data = {
                        "scm_id": scm_id,
                        "model": model,
                        "explainer": explainer,
                        "seed": seed,
                        "n_explained_rows": n_explain,
                        "mean_abs_attributions": mean_abs,
                        "compute_time_seconds": round(float(rng.uniform(0.1, 1.5)), 3),
                        "config": {"model": model, "explainer": explainer},
                    }
                    with open(sidecar_file, "w", encoding="utf-8") as f:
                        json.dump(sidecar_data, f, indent=2)

                    # 2. Add to errors.csv
                    # Decompose: total_error = estimand_gap + model_misfit + explainer_error
                    model_misfit_val = float(rng.uniform(0.05, 0.15))
                    explainer_err_val = float(rng.uniform(0.01, 0.08))
                    total_l2 = estimand_gap_val + model_misfit_val + explainer_err_val
                    total_rank = float(rng.uniform(0.1, 0.5))

                    error_rows.append({
                        "scm_id": scm_id,
                        "model": model,
                        "explainer": explainer,
                        "seed": seed,
                        "total_error_l2": round(total_l2, 4),
                        "total_error_rank": round(total_rank, 4),
                        "estimand_gap": round(estimand_gap_val, 4),
                        "model_misfit": round(model_misfit_val, 4),
                        "explainer_error": round(explainer_err_val, 4),
                    })

                    # 3. Add to observables.csv
                    observable_rows.append({
                        "scm_id": scm_id,
                        "model": model,
                        "explainer": explainer,
                        "seed": seed,
                        "max_vif": round(float(rng.uniform(1.2, 5.0)), 3),
                        "mean_vif": round(float(rng.uniform(1.1, 2.5)), 3),
                        "mutual_info_max": round(float(rng.uniform(0.2, 0.8)), 3),
                        "mutual_info_mean": round(float(rng.uniform(0.1, 0.4)), 3),
                        "test_mse": round(float(rng.uniform(0.1, 0.6)), 3),
                        "test_r2": round(float(rng.uniform(0.65, 0.95)), 3),
                        "explainer_stability": round(float(rng.uniform(0.85, 0.99)), 3),
                        "explainer_sensitivity": round(float(rng.uniform(0.05, 0.35)), 3),
                        "prediction_concentration": round(float(rng.uniform(0.2, 0.6)), 3),
                        "explainer_disagreement_rank": round(float(rng.uniform(0.05, 0.35)), 3),
                        "explainer_disagreement_l2": round(float(rng.uniform(0.1, 0.4)), 3),
                        "n_train": 1200,
                    })

    # Save CSVs
    errors_df = pd.DataFrame(error_rows)
    errors_csv = res_path / "errors.csv"
    errors_df.to_csv(errors_csv, index=False)

    obs_df = pd.DataFrame(observable_rows)
    obs_csv = res_path / "observables.csv"
    obs_df.to_csv(obs_csv, index=False)

    print(f"Mock generation complete!")
    print(f"  - Attributions: {len(error_rows)} parquet/JSON pairs in {attr_dir}")
    print(f"  - Errors CSV: {errors_csv} ({len(errors_df)} rows)")
    print(f"  - Observables CSV: {obs_csv} ({len(obs_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock results conforming to SCHEMAS.md.")
    parser.add_argument(
        "--scms-dir",
        type=str,
        default="data/scms_small",
        help="Path to SCM benchmark directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to output results directory.",
    )
    args = parser.parse_args()

    make_mock_results(args.scms_dir, args.results_dir)


if __name__ == "__main__":
    main()
