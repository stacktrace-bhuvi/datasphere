"""CLI and programmatic generator for synthetic SCM benchmark datasets."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.scm.scm import SCM


def generate_benchmark_from_config(config: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    """Generate a suite of SCMs and write individual JSON files and a manifest.csv."""
    out_path = Path(output_dir)
    scms_dir = out_path / "scms"
    scms_dir.mkdir(parents=True, exist_ok=True)

    n_per_family = int(config.get("n_scms_per_family", 20))
    d_min, d_max = config.get("d_range", [5, 10])
    base_seed = int(config.get("base_seed", 42))
    nonlin_prob = float(config.get("nonlinearity_prob", 0.5))
    inter_prob = float(config.get("interaction_prob", 0.2))
    noise_type = str(config.get("noise_type", "gaussian"))
    noise_scale = float(config.get("noise_scale", 1.0))
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

    manifest_rows: list[dict[str, Any]] = []
    global_idx = 0

    for f_idx, family in enumerate(families):
        split = "heldout" if family in heldout_families else "train"

        for i in range(n_per_family):
            d = d_min + ((i + f_idx) % (d_max - d_min + 1))
            seed = base_seed + (f_idx * 10_000) + i
            scm_id = f"scm_{global_idx:04d}_{family}_d{d}"

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

            # Save SCM JSON
            json_file = scms_dir / f"{scm_id}.json"
            scm.to_json(json_file)

            # Metadata for manifest
            manifest_rows.append({
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
            })
            global_idx += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = out_path / "manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Generated {len(manifest_df)} SCMs in {scms_dir}")
    print(f"Manifest written to {manifest_csv_path}")

    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SCM benchmark dataset.")
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
        help="Output directory for SCM JSON files and manifest.csv.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generate_benchmark_from_config(config, args.output_dir)


if __name__ == "__main__":
    main()
