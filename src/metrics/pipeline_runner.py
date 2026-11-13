"""Build `results/errors.csv` from saved ground-truth and attribution artifacts.

Inputs (see docs/SCHEMAS.md):
  <scms_dir>/<scm_id>/ground_truth.json     tau, oracle_mean_abs_shap, explain/background_row_indices
  <attributions_dir>/<prefix>.json + .parquet   written by scripts/run_attributions.py

Each attribution sidecar must carry `fitted_model_mean_abs_shap` (exact SHAP of the fitted model,
computed independently of the explainer; see src/explainers/reference.py) plus its `reference`
provenance block. Sidecars lacking it are rejected with instructions -- never filled in.

Everything is validated before any number is computed: filename/content identity, vector lengths
and finiteness, feature order, that the reference was computed on the SAME explain/background rows
as the ground truth, and that the sidecar agrees with its parquet.

Global explainers (permutation importance: increase in test MSE, not SHAP units) cannot be compared
to SHAP-unit magnitudes by L2. For them `total_error_l2` and `explainer_error` are written as NaN,
while the scale-free `total_error_rank` and the explainer-independent `estimand_gap` / `model_misfit`
are still reported.

Every row also carries scale-free columns: the same decomposition on importance shares
(`*_share`, |v| / sum|v|), on ranks (`*_rank`, 1 - Kendall tau-b between consecutive stages)
and `top_k_overlap`. These are valid for global explainers as well.

Output is a pure function of the input artifacts: sorted, deterministic, written atomically,
so re-running never duplicates rows or loses valid ones.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.explainers.reference import MissingReferenceError, feature_names_for, read_reference
from src.metrics.decomposition import decompose_error
from src.metrics.error_metrics import normalized_l2, rank_metrics

ERROR_COLUMNS = ["scm_id", "model", "explainer", "seed", "total_error_l2",
                 "total_error_rank", "estimand_gap", "model_misfit", "explainer_error",
                 # scale-free extensions (importance shares and ranks; valid for global explainers too)
                 "total_error_share_l2", "estimand_gap_share", "model_misfit_share", "explainer_error_share",
                 "estimand_gap_rank", "model_misfit_rank", "explainer_error_rank", "top_k_overlap"]
SCALE_FREE_COLUMNS = ERROR_COLUMNS[9:]
KEY_COLUMNS = ["scm_id", "model", "explainer", "seed"]
REQUIRED_SIDECAR = ["scm_id", "model", "explainer", "seed", "n_explained_rows", "mean_abs_attributions"]
PARQUET_RTOL = 1e-6


class PipelineInputError(ValueError):
    """One or more artifacts are missing or invalid (message lists them)."""


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        raise PipelineInputError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(obj, dict):
        raise PipelineInputError(f"{path}: expected a JSON object")
    return obj


def _vector(obj: dict, key: str, source: str, d: int | None = None) -> np.ndarray:
    if key not in obj:
        raise PipelineInputError(f"{source}: missing '{key}'")
    try:
        v = np.asarray(obj[key], dtype=float)
    except (TypeError, ValueError) as e:
        raise PipelineInputError(f"{source}: '{key}' is not a numeric vector") from e
    if v.ndim != 1 or v.size == 0:
        raise PipelineInputError(f"{source}: '{key}' must be a non-empty 1-D vector")
    if not np.all(np.isfinite(v)):
        raise PipelineInputError(f"{source}: '{key}' contains non-finite values")
    if np.any(v < 0):
        raise PipelineInputError(f"{source}: '{key}' must be non-negative magnitudes")
    if d is not None and v.size != d:
        raise PipelineInputError(f"{source}: '{key}' has length {v.size}, expected d={d}")
    return v


def load_ground_truth(scms_dir: Path, scm_id: str) -> dict:
    """Validated ground truth of one SCM (raises PipelineInputError)."""
    path = scms_dir / scm_id / "ground_truth.json"
    if not path.exists():
        raise PipelineInputError(
            f"missing {path} (generate SCMs first: `python tasks.py generate-small|generate-full`)")
    gt = _load_json(path)
    tau = _vector(gt, "tau", str(path))
    d = tau.size
    _vector(gt, "oracle_mean_abs_shap", str(path), d)
    for key in ("explain_row_indices", "background_row_indices"):
        if not isinstance(gt.get(key), list) or not gt[key]:
            raise PipelineInputError(f"{path}: '{key}' must be a non-empty list")
    if gt.get("scm_id", scm_id) != scm_id:
        raise PipelineInputError(f"{path}: scm_id {gt.get('scm_id')!r} != directory name {scm_id!r}")
    if "d" in gt and int(gt["d"]) != d:
        raise PipelineInputError(f"{path}: d={gt['d']} but tau has length {d}")
    return gt


def _evaluate_sidecar(sidecar_path: Path, gt_cache: dict, scms_dir: Path) -> dict:
    src = sidecar_path.name
    attr = _load_json(sidecar_path)
    absent = [k for k in REQUIRED_SIDECAR if k not in attr]
    if absent:
        raise PipelineInputError(f"{src}: missing required fields {absent}")
    scm_id = attr["scm_id"]
    expected = f"{scm_id}__{attr['model']}__{attr['explainer']}__seed{int(attr['seed'])}"
    if sidecar_path.stem != expected:
        raise PipelineInputError(f"{src}: filename does not match its contents (expected {expected}.json)")
    parquet = sidecar_path.with_suffix(".parquet")
    if not parquet.exists():
        raise PipelineInputError(f"{src}: companion {parquet.name} is missing")

    if scm_id not in gt_cache:
        gt_cache[scm_id] = load_ground_truth(scms_dir, scm_id)
    gt = gt_cache[scm_id]
    tau = np.asarray(gt["tau"], float)
    d = tau.size
    oracle = np.asarray(gt["oracle_mean_abs_shap"], float)

    empirical = _vector(attr, "mean_abs_attributions", src, d)
    fitted, meta = read_reference(attr, src)
    if fitted.size != d:
        raise PipelineInputError(f"{src}: 'fitted_model_mean_abs_shap' has length {fitted.size}, expected d={d}")
    if meta.get("feature_names") != feature_names_for(d):
        raise PipelineInputError(f"{src}: reference feature order {meta.get('feature_names')} != X1..X{d}")
    if meta.get("explain_row_indices") != gt["explain_row_indices"] \
            or meta.get("background_row_indices") != gt["background_row_indices"]:
        raise PipelineInputError(
            f"{src}: reference SHAP was computed on different explain/background rows than "
            f"{scm_id}/ground_truth.json; regenerate with scripts/run_attributions.py --force")
    if int(attr["n_explained_rows"]) != len(gt["explain_row_indices"]):
        raise PipelineInputError(f"{src}: n_explained_rows={attr['n_explained_rows']} != "
                                 f"{len(gt['explain_row_indices'])} ground-truth explain rows")

    cols = feature_names_for(d)
    df = pd.read_parquet(parquet)
    if list(df.columns) != ["row_index", *cols, "base_value"]:
        raise PipelineInputError(f"{parquet.name}: unexpected columns {list(df.columns)}")
    if df["row_index"].tolist() != gt["explain_row_indices"]:
        raise PipelineInputError(f"{parquet.name}: row_index does not match ground-truth explain rows")
    from_parquet = np.abs(df[cols].to_numpy(float)).mean(axis=0)
    if not np.allclose(from_parquet, empirical, rtol=PARQUET_RTOL, atol=1e-12):
        raise PipelineInputError(f"{src}: mean_abs_attributions disagrees with {parquet.name}")

    is_global = bool(attr.get("is_global", False))
    r = decompose_error(tau, oracle, fitted, empirical)
    out_l2, rank, gap = r["total_error_l2"], r["total_error_rank"], r["estimand_gap"]
    misfit, out_expl = r["model_misfit"], r["explainer_error"]
    if is_global:
        # raw L2 between MSE-unit importances and SHAP-unit magnitudes is meaningless
        out_l2 = out_expl = math.nan
    row = {"scm_id": scm_id, "model": attr["model"], "explainer": attr["explainer"],
           "seed": int(attr["seed"]), "total_error_l2": out_l2, "total_error_rank": rank,
           "estimand_gap": gap, "model_misfit": misfit, "explainer_error": out_expl}
    row.update({k: r[k] for k in SCALE_FREE_COLUMNS})
    return row


def build_error_table(scms_dir: str | Path, attributions_dir: str | Path,
                      output_csv: str | Path, strict: bool = True) -> pd.DataFrame:
    """Compute and write errors.csv.

    strict=True (default): any invalid artifact aborts with a message listing the problems and
    nothing is written. strict=False: invalid artifacts are skipped and listed in
    `<output dir>/errors_skipped.csv`; valid rows are still written.
    """
    scms_dir, attributions_dir = Path(scms_dir), Path(attributions_dir)
    if not attributions_dir.is_dir():
        raise PipelineInputError(f"attributions directory not found: {attributions_dir} "
                                 "(run scripts/run_attributions.py first)")
    sidecars = sorted(attributions_dir.glob("*.json"))
    if not sidecars:
        raise PipelineInputError(f"no attribution sidecars (*.json) in {attributions_dir}")

    rows, problems, gt_cache, seen = [], [], {}, {}
    for sc in sidecars:
        try:
            row = _evaluate_sidecar(sc, gt_cache, scms_dir)
            key = tuple(row[k] for k in KEY_COLUMNS)
            if key in seen:
                raise PipelineInputError(f"{sc.name}: duplicate record for {key} (also {seen[key]})")
            seen[key] = sc.name
            rows.append(row)
        except (PipelineInputError, MissingReferenceError) as e:
            problems.append((sc.name, str(e)))

    out = Path(output_csv)
    if problems and strict:
        shown = "\n- ".join(m for _, m in problems[:20])
        more = f"\n... ({len(problems) - 20} more)" if len(problems) > 20 else ""
        raise PipelineInputError(f"{len(problems)} of {len(sidecars)} attribution sidecars cannot be "
                                 f"evaluated:\n- {shown}{more}")
    if not rows:
        raise PipelineInputError(f"no valid attribution sidecars in {attributions_dir}")

    df = pd.DataFrame(rows, columns=ERROR_COLUMNS).sort_values(KEY_COLUMNS).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    if problems:
        pd.DataFrame(problems, columns=["sidecar", "reason"]).to_csv(out.parent / "errors_skipped.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scms-dir", required=True)
    parser.add_argument("--attributions-dir", required=True)
    parser.add_argument("--output-csv", default="results/errors.csv")
    parser.add_argument("--allow-skipped", action="store_true",
                        help="skip invalid sidecars (listed in errors_skipped.csv) instead of aborting")
    args = parser.parse_args()
    df = build_error_table(args.scms_dir, args.attributions_dir, args.output_csv, strict=not args.allow_skipped)
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
