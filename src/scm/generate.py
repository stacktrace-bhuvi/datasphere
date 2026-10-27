"""CLI and programmatic generator for synthetic SCM benchmark datasets with ground truth."""

from __future__ import annotations
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from src.scm.scm import SCM
from src.scm.ground_truth import compute_ground_truth


def _generate_single_scm(
    args_tuple: tuple[dict[str, Any], Path, bool]
) -> dict[str, Any]:
    """Worker function to generate a single SCM directory and all its artifacts."""
    spec, output_dir, force = args_tuple

    scm_id = spec["scm_id"]
    family = spec["family"]
    d = spec["d"]
    seed = spec["seed"]
    split = spec["split"]
    n_samples = spec["n_samples"]
    nonlin_prob = spec["nonlinearity_prob"]
    noise_type = spec["noise_type"]
    noise_scale = spec["noise_scale"]
    inter_prob = spec["interaction_prob"]
    standardize = spec["standardize"]
    clip_bound = spec["clip_bound"]

    scm_dir = output_dir / scm_id
    scm_dir.mkdir(parents=True, exist_ok=True)

    data_path = scm_dir / "data.parquet"
    scm_path = scm_dir / "scm.json"
    meta_path = scm_dir / "meta.json"
    gt_path = scm_dir / "ground_truth.json"

    # Check if all files exist and skip if not forced
    all_exist = data_path.exists() and scm_path.exists() and meta_path.exists() and gt_path.exists()
    if all_exist and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        return {
            "scm_id": scm_id,
            "family": family,
            "d": d,
            "n_nonlinear": meta.get("n_nonlinear", 0),
            "noise_type": noise_type,
            "noise_scale": noise_scale,
            "has_confounding": meta.get("has_confounding", False),
            "mediation_share": meta.get("mediation_share", 0.0),
            "standardize": standardize,
            "clip_bound": clip_bound,
            "clipped_fraction_Y": gt.get("clipped_fraction_Y", 0.0),
            "seed": seed,
            "split": split,
            "skipped": True,
        }

    # 1. Build SCM
    scm = SCM.build(
        family=family,
        d=d,
        seed=seed,
        nonlinearity_prob=nonlin_prob,
        noise_type=noise_type,
        noise_scale=noise_scale,
        interaction_prob=inter_prob,
        standardize=standardize,
        clip_bound=clip_bound,
    )
    scm.metadata["scm_id"] = scm_id
    scm.metadata["split"] = split

    # 2. Sample observational data and create 60/20/20 train/val/test split
    df_obs = scm.sample(n=n_samples, seed=seed)

    n_train = int(0.60 * n_samples)
    n_val = int(0.20 * n_samples)
    n_test = n_samples - n_train - n_val

    splits_array = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    split_rng = np.random.default_rng(seed + 999)
    split_rng.shuffle(splits_array)
    df_obs["row_split"] = splits_array

    # Save data.parquet
    df_obs.to_parquet(data_path, index=False)

    # Save scm.json
    scm.to_json(scm_path)

    # 3. Compute ground truth and estimand gap metrics
    gt_dict = compute_ground_truth(
        scm=scm,
        df=df_obs,
        n_repeats=5,
        n_ref=50,
        n_noise_per_ref=5_000,
        zero_tol=1e-12,
        n_explain=100,
        n_background=50,
    )
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f, indent=2)

    # 4. Save meta.json
    meta_dict = {
        "scm_id": scm_id,
        "family": family,
        "d": d,
        "n_nonlinear": scm.metadata.get("n_nonlinear", 0),
        "noise_type": noise_type,
        "noise_scale": noise_scale,
        "has_confounding": scm.metadata.get("has_confounding", False),
        "mediation_share": scm.metadata.get("mediation_share", 0.0),
        "standardize": standardize,
        "clip_bound": clip_bound,
        "seed": seed,
        "split": split,
        "n_samples": n_samples,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return {
        "scm_id": scm_id,
        "family": family,
        "d": d,
        "n_nonlinear": meta_dict["n_nonlinear"],
        "noise_type": noise_type,
        "noise_scale": noise_scale,
        "has_confounding": meta_dict["has_confounding"],
        "mediation_share": meta_dict["mediation_share"],
        "standardize": standardize,
        "clip_bound": clip_bound,
        "clipped_fraction_Y": gt_dict["clipped_fraction_Y"],
        "seed": seed,
        "split": split,
        "skipped": False,
    }


def generate_benchmark_from_config(
    config: dict[str, Any],
    output_dir: str | Path,
    workers: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    """Generate a complete SCM benchmark suite with parquet datasets and ground truth."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_per_family = int(config.get("n_scms_per_family", 20))
    d_min, d_max = config.get("d_range", [5, 10])
    base_seed = int(config.get("base_seed", 42))
    nonlin_prob = float(config.get("nonlinearity_prob", 0.5))
    inter_prob = float(config.get("interaction_prob", 0.2))
    noise_type = str(config.get("noise_type", "gaussian"))
    noise_scale = float(config.get("noise_scale", 1.0))
    n_samples = int(config.get("n_samples", 2000))
    standardize = bool(config.get("standardize", True))
    clip_bound = float(config["clip_bound"]) if config.get("clip_bound") is not None else 10.0

    families = config.get("families", [
        "chain_mediation",
        "confounded_fork",
        "feature_collider",
        "erdos_renyi_dag",
        "layered",
    ])

    splits_cfg = config.get("splits", {})
    heldout_families = set(splits_cfg.get("heldout_families", ["layered"]))

    specs: list[dict[str, Any]] = []
    global_idx = 0

    for f_idx, family in enumerate(families):
        split = "heldout" if family in heldout_families else "train"

        for i in range(n_per_family):
            d = d_min + ((i + f_idx) % (d_max - d_min + 1))
            seed = base_seed + (f_idx * 10_000) + i
            scm_id = f"scm_{global_idx:04d}_{family}_d{d}"

            spec = {
                "scm_id": scm_id,
                "family": family,
                "d": d,
                "seed": seed,
                "split": split,
                "n_samples": n_samples,
                "nonlinearity_prob": nonlin_prob,
                "noise_type": noise_type,
                "noise_scale": noise_scale,
                "interaction_prob": inter_prob,
                "standardize": standardize,
                "clip_bound": clip_bound,
            }
            specs.append(spec)
            global_idx += 1

    total_scms = len(specs)
    print(f"Preparing {total_scms} SCMs across {len(families)} families into {out_path} (workers={workers}, force={force})...")

    results: list[dict[str, Any]] = []

    # Prepare worker arguments
    worker_args = [(spec, out_path, force) for spec in specs]

    if workers <= 1:
        for idx, item in enumerate(worker_args, 1):
            res = _generate_single_scm(item)
            results.append(res)
            action = "Skipped (cached)" if res.get("skipped") else "Generated"
            print(f"[{idx}/{total_scms}] {action}: {res['scm_id']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_single_scm, item) for item in worker_args]
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1
                action = "Skipped (cached)" if res.get("skipped") else "Generated"
                print(f"[{completed}/{total_scms}] {action}: {res['scm_id']}")

    # Sort results by scm_id to maintain deterministic ordering
    results.sort(key=lambda r: r["scm_id"])

    manifest_rows = []
    excessive_clipping_scms: list[tuple[str, float]] = []

    for r in results:
        clip_y = float(r.get("clipped_fraction_Y", 0.0))
        if clip_y > 0.005:  # 0.5% threshold
            excessive_clipping_scms.append((r["scm_id"], clip_y))

        manifest_rows.append({
            "scm_id": r["scm_id"],
            "family": r["family"],
            "d": r["d"],
            "n_nonlinear": r["n_nonlinear"],
            "noise_type": r["noise_type"],
            "noise_scale": r["noise_scale"],
            "has_confounding": r["has_confounding"],
            "mediation_share": r["mediation_share"],
            "standardize": r["standardize"],
            "clip_bound": r["clip_bound"],
            "clipped_fraction_Y": clip_y,
            "seed": r["seed"],
            "split": r["split"],
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = out_path / "manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)

    print(f"\nManifest written to {manifest_csv_path}")

    # Check and warn if any SCM exceeds 0.5% clipping on Y
    if excessive_clipping_scms:
        msg = (
            f"WARNING: {len(excessive_clipping_scms)} SCMs exceeded the 0.5% clipping threshold for Y:\n"
            + "\n".join(f"  - {sid}: {frac * 100:.2f}% clipped" for sid, frac in excessive_clipping_scms[:5])
        )
        if len(excessive_clipping_scms) > 5:
            msg += f"\n  ... and {len(excessive_clipping_scms) - 5} more."
        warnings.warn(msg, UserWarning)
        print(msg)

    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SCM benchmark dataset with causal ground truth.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/scm_default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/scms",
        help="Output directory for SCM directories and manifest.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing SCM outputs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generate_benchmark_from_config(config, args.output_dir, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
