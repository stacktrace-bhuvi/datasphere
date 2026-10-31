"""Build real error tables from saved ground-truth and attribution artifacts.

Expected per-run JSON sidecar fields:
  scm_id, model, explainer, seed, mean_abs_attributions,
  fitted_model_mean_abs_shap

The latter must be a high-accuracy/reference SHAP computation on the fitted model,
not copied from the empirical explainer output. Mock artifacts are not accepted
as research results unless --allow-mock is explicitly provided (no mock flag is
implemented here; use real artifacts only).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from src.metrics.decomposition import decompose_error


def build_error_table(scms_dir: str | Path, attributions_dir: str | Path,
                      output_csv: str | Path) -> pd.DataFrame:
    scms_dir, attributions_dir = Path(scms_dir), Path(attributions_dir)
    rows = []
    missing = []
    for sidecar in sorted(attributions_dir.glob("*.json")):
        with sidecar.open(encoding="utf-8") as f:
            attr = json.load(f)
        scm_id = attr.get("scm_id")
        gt_path = scms_dir / scm_id / "ground_truth.json"
        if not gt_path.exists():
            missing.append(f"{sidecar.name}: missing {gt_path}")
            continue
        required = ["mean_abs_attributions", "fitted_model_mean_abs_shap",
                    "model", "explainer", "seed"]
        absent = [k for k in required if k not in attr]
        if absent:
            missing.append(
                f"{sidecar.name}: missing {absent}; cannot compute a valid "
                "three-way decomposition without fitted_model_mean_abs_shap"
            )
            continue
        with gt_path.open(encoding="utf-8") as f:
            gt = json.load(f)
        # tau is the benchmark's absolute total-effect vector; oracle SHAP is
        # mean absolute attribution. Keep all quantities in magnitude space.
        result = decompose_error(
            gt["tau"],
            gt["oracle_mean_abs_shap"],
            attr["fitted_model_mean_abs_shap"],
            attr["mean_abs_attributions"],
        )
        rows.append({
            "scm_id": scm_id,
            "model": attr["model"],
            "explainer": attr["explainer"],
            "seed": int(attr["seed"]),
            "total_error_l2": result["total_error_l2"],
            "total_error_rank": result["total_error_rank"],
            "estimand_gap": result["estimand_gap"],
            "model_misfit": result["model_misfit"],
            "explainer_error": result["explainer_error"],
        })
    if missing:
        raise ValueError(
            "Some attribution sidecars cannot be evaluated:\n- " + "\n- ".join(missing[:20])
            + ("\n... (more omitted)" if len(missing) > 20 else "")
        )
    if not rows:
        raise ValueError(f"No usable attribution sidecars found in {attributions_dir}")
    df = pd.DataFrame(rows, columns=[
        "scm_id", "model", "explainer", "seed", "total_error_l2",
        "total_error_rank", "estimand_gap", "model_misfit", "explainer_error"
    ])
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scms-dir", required=True)
    parser.add_argument("--attributions-dir", required=True)
    parser.add_argument("--output-csv", default="results/errors.csv")
    args = parser.parse_args()
    df = build_error_table(args.scms_dir, args.attributions_dir, args.output_csv)
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
