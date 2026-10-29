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
    args_tuple: tuple[dict[str, Any], Path, bool, bool]
) -> dict[str, Any]:
    """Worker function to generate or dry-run a single SCM directory."""
    spec, output_dir, force, dry_run = args_tuple

    scm_id = spec["scm_id"]
    family = spec["family"]
    d = spec["d"]
    initial_seed = spec["seed"]
    split = spec["split"]
    n_samples = spec["n_samples"]
    nonlin_prob = spec["nonlinearity_prob"]
    noise_type = spec["noise_type"]
    noise_scale = spec["noise_scale"]
    inter_prob = spec["interaction_prob"]
    standardize = spec["standardize"]
    clip_bound = spec["clip_bound"]

    scm_dir = output_dir / scm_id
    data_path = scm_dir / "data.parquet"
    scm_path = scm_dir / "scm.json"
    meta_path = scm_dir / "meta.json"
    gt_path = scm_dir / "ground_truth.json"

    # Check if all files exist and skip if not forced (only if not dry_run)
    if not dry_run:
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
                "max_node_clip": max(gt.get("clipped_fractions", {}).values()) if gt.get("clipped_fractions") else 0.0,
                "seed": meta.get("seed", initial_seed),
                "attempts": meta.get("attempts", 1),
                "split": split,
                "skipped": True,
            }

    # 1. Rejection sampling loop (max 20 attempts)
    max_attempts = 20
    current_seed = initial_seed
    accepted = False
    scm = None
    df_obs = None
    clip_fracs: dict[str, float] = {}
    clip_y = 0.0
    max_node_clip = 0.0
    attempt = 1

    feature_cols = [f"X{i}" for i in range(1, d + 1)]
    threshold = (float(clip_bound) - 1e-6) if clip_bound is not None else 1e9

    for attempt in range(1, max_attempts + 1):
        scm = SCM.build(
            family=family,
            d=d,
            seed=current_seed,
            nonlinearity_prob=nonlin_prob,
            noise_type=noise_type,
            noise_scale=noise_scale,
            interaction_prob=inter_prob,
            standardize=standardize,
            clip_bound=clip_bound,
        )
        scm.metadata["scm_id"] = scm_id
        scm.metadata["split"] = split
        scm.metadata["seed"] = current_seed
        scm.metadata["clip_bound"] = clip_bound
        scm.metadata["standardize"] = standardize

        df_obs = scm.sample(n=n_samples, seed=current_seed)

        if clip_bound is not None:
            clip_fracs = {}
            for col in feature_cols + ["Y"]:
                vals = np.abs(df_obs[col].to_numpy(dtype=np.float64))
                clip_fracs[col] = float(np.mean(vals >= threshold))
            clip_y = clip_fracs.get("Y", 0.0)
            max_node_clip = max(clip_fracs.values())
        else:
            clip_fracs = {c: 0.0 for c in feature_cols + ["Y"]}
            clip_y = 0.0
            max_node_clip = 0.0

        # Acceptance threshold: clipped_fraction_Y <= 0.5% and any node <= 1.0%
        if clip_y <= 0.005 and max_node_clip <= 0.01:
            accepted = True
            break
        else:
            if attempt < max_attempts:
                current_seed += 1

    if not accepted:
        warnings.warn(
            f"SCM {scm_id} reached max {max_attempts} attempts without satisfying clipping thresholds "
            f"(clip_Y={clip_y*100:.2f}%, max_clip={max_node_clip*100:.2f}%)",
            UserWarning,
        )

    if dry_run:
        return {
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
            "clipped_fraction_Y": clip_y,
            "max_node_clip": max_node_clip,
            "seed": current_seed,
            "attempts": attempt,
            "split": split,
            "skipped": False,
        }

    # 2. Observational data 60/20/20 train/val/test split and persistence
    scm_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(0.60 * n_samples)
    n_val = int(0.20 * n_samples)
    n_test = n_samples - n_train - n_val

    splits_array = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    split_rng = np.random.default_rng(current_seed + 999)
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
        "seed": current_seed,
        "attempts": attempt,
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
        "max_node_clip": max_node_clip,
        "seed": current_seed,
        "attempts": attempt,
        "split": split,
        "skipped": False,
    }


def generate_benchmark_from_config(
    config: dict[str, Any],
    output_dir: str | Path,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Generate a complete SCM benchmark suite with parquet datasets and ground truth (or dry-run)."""
    out_path = Path(output_dir)
    if not dry_run:
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
    mode_str = "DRY-RUN (no disk writes)" if dry_run else f"writing into {out_path}"
    print(f"Executing {total_scms} SCMs across {len(families)} families [{mode_str}] (workers={workers}, force={force})...")

    results: list[dict[str, Any]] = []

    # Prepare worker arguments
    worker_args = [(spec, out_path, force, dry_run) for spec in specs]

    if workers <= 1:
        for idx, item in enumerate(worker_args, 1):
            res = _generate_single_scm(item)
            results.append(res)
            action = "Evaluated" if dry_run else ("Skipped (cached)" if res.get("skipped") else "Generated")
            print(f"[{idx}/{total_scms}] {action}: {res['scm_id']} (attempts={res['attempts']}, clip_Y={res['clipped_fraction_Y']*100:.2f}%)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_single_scm, item) for item in worker_args]
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1
                action = "Evaluated" if dry_run else ("Skipped (cached)" if res.get("skipped") else "Generated")
                print(f"[{completed}/{total_scms}] {action}: {res['scm_id']} (attempts={res['attempts']}, clip_Y={res['clipped_fraction_Y']*100:.2f}%)")

    # Sort results by scm_id to maintain deterministic ordering
    results.sort(key=lambda r: r["scm_id"])

    manifest_rows = []
    excessive_clipping_scms: list[tuple[str, float, float]] = []

    for r in results:
        clip_y = float(r.get("clipped_fraction_Y", 0.0))
        max_clip = float(r.get("max_node_clip", clip_y))
        if clip_y > 0.005 or max_clip > 0.01:
            excessive_clipping_scms.append((r["scm_id"], clip_y, max_clip))

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
            "max_node_clip": max_clip,
            "seed": r["seed"],
            "attempts": r["attempts"],
            "split": r["split"],
        })

    manifest_df = pd.DataFrame(manifest_rows)

    # Summary reporting
    clip_y_vals = manifest_df["clipped_fraction_Y"].to_numpy()
    max_node_vals = manifest_df["max_node_clip"].to_numpy()

    print("\n" + "=" * 60)
    print("CLIPPED-FRACTION DISTRIBUTION:")
    print(f"  Outcome Y:")
    print(f"    Min:             {np.min(clip_y_vals)*100:.4f}%")
    print(f"    25th Percentile: {np.percentile(clip_y_vals, 25)*100:.4f}%")
    print(f"    50th Percentile: {np.percentile(clip_y_vals, 50)*100:.4f}%")
    print(f"    75th Percentile: {np.percentile(clip_y_vals, 75)*100:.4f}%")
    print(f"    95th Percentile: {np.percentile(clip_y_vals, 95)*100:.4f}%")
    print(f"    Max:             {np.max(clip_y_vals)*100:.4f}%")
    print(f"    Zero Clipping:   {(clip_y_vals == 0.0).mean()*100:.2f}% of SCMs")
    print(f"  Max Node (any feature or Y):")
    print(f"    Min:             {np.min(max_node_vals)*100:.4f}%")
    print(f"    50th Percentile: {np.percentile(max_node_vals, 50)*100:.4f}%")
    print(f"    95th Percentile: {np.percentile(max_node_vals, 95)*100:.4f}%")
    print(f"    Max:             {np.max(max_node_vals)*100:.4f}%")
    print(f"    Zero Clipping:   {(max_node_vals == 0.0).mean()*100:.2f}% of SCMs")

    print("\nREJECTION COUNT PER FAMILY:")
    print(f"  {'Family':<20} {'SCMs':<6} {'Total Attempts':<16} {'Rejections':<12} {'Max Attempts':<12}")
    print("  " + "-" * 68)
    for fam, group in manifest_df.groupby("family"):
        n_scms = len(group)
        tot_attempts = int(group["attempts"].sum())
        rejections = int(tot_attempts - n_scms)
        max_att = int(group["attempts"].max())
        print(f"  {fam:<20} {n_scms:<6} {tot_attempts:<16} {rejections:<12} {max_att:<12}")
    print("=" * 60 + "\n")

    if not dry_run:
        manifest_csv_path = out_path / "manifest.csv"
        # Drop temporary max_node_clip column if desired or keep it
        save_cols = [c for c in manifest_df.columns if c != "max_node_clip"]
        manifest_df[save_cols].to_csv(manifest_csv_path, index=False)
        print(f"Manifest written to {manifest_csv_path}")

    # Check and warn if any accepted SCM exceeds thresholds
    if excessive_clipping_scms:
        msg = (
            f"WARNING: {len(excessive_clipping_scms)} SCMs exceeded clipping thresholds:\n"
            + "\n".join(f"  - {sid}: Y={cy*100:.2f}%, max_node={mn*100:.2f}%" for sid, cy, mn in excessive_clipping_scms[:5])
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute SCMs and clipping statistics without writing data or ground truth to disk.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generate_benchmark_from_config(
        config,
        args.output_dir,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
