"""Build the real error table (results/errors.csv) from saved ground-truth and attribution artifacts.

Stage 2 (scripts/run_attributions.py) writes one sidecar per (scm, model, explainer, seed).
The ``exact`` explainer is brute-force interventional SHAP of the FITTED model, i.e. the
Phi(f) term of the decomposition. For every other explainer on the same (scm, model, seed)
we read Phi(f) from that ``exact`` sidecar, so the three-way decomposition

    tau  ->  Phi(f*)  ->  Phi(f)  ->  E_hat(f)
         gap        misfit       explainer error

uses real quantities only. A sidecar may also carry ``fitted_model_mean_abs_shap``
directly; that takes precedence when present. Rows without a matching ``exact`` sidecar
are skipped and reported (use --strict to fail instead).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from src.metrics.decomposition import decompose_error

ERROR_COLUMNS = [
    "scm_id", "model", "explainer", "seed",
    "total_error_l2", "total_error_rank",
    "estimand_gap", "model_misfit", "explainer_error",
    "total_error_share_l2", "estimand_gap_share", "model_misfit_share", "explainer_error_share",
    "estimand_gap_rank", "model_misfit_rank", "explainer_error_rank",
    "top_k_overlap",
]


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_error_table(scms_dir: str | Path, attributions_dir: str | Path,
                      output_csv: str | Path | None, strict: bool = False) -> pd.DataFrame:
    scms_dir, attributions_dir = Path(scms_dir), Path(attributions_dir)
    sidecars = {p.stem: _load_json(p) for p in sorted(attributions_dir.glob("*.json"))}

    # Index the fitted-model exact SHAP by (scm_id, model, seed).
    fitted_ref: dict[tuple[str, str, int], list[float]] = {}
    for attr in sidecars.values():
        if attr.get("explainer") == "exact":
            fitted_ref[(attr["scm_id"], attr["model"], int(attr["seed"]))] = attr["mean_abs_attributions"]

    rows, problems = [], []
    gt_cache: dict[str, dict] = {}
    for name, attr in sidecars.items():
        required = ["scm_id", "model", "explainer", "seed", "mean_abs_attributions"]
        absent = [k for k in required if k not in attr]
        if absent:
            problems.append(f"{name}.json: missing {absent}")
            continue
        scm_id, model, explainer, seed = attr["scm_id"], attr["model"], attr["explainer"], int(attr["seed"])

        if scm_id not in gt_cache:
            gt_path = scms_dir / scm_id / "ground_truth.json"
            if not gt_path.exists():
                problems.append(f"{name}.json: missing {gt_path}")
                continue
            gt_cache[scm_id] = _load_json(gt_path)
        gt = gt_cache[scm_id]

        fitted = attr.get("fitted_model_mean_abs_shap") or fitted_ref.get((scm_id, model, seed))
        if fitted is None:
            problems.append(f"{name}.json: no `exact` sidecar for ({scm_id}, {model}, seed {seed}); "
                            "run scripts/run_attributions.py with --explainers exact ...")
            continue

        # tau is the benchmark's absolute total-effect vector; oracle SHAP is mean
        # absolute attribution. Keep all quantities in magnitude space.
        res = decompose_error(gt["tau"], gt["oracle_mean_abs_shap"], fitted, attr["mean_abs_attributions"])
        rows.append({"scm_id": scm_id, "model": model, "explainer": explainer, "seed": seed,
                     **{k: res[k] for k in ERROR_COLUMNS[4:]}})

    if problems:
        msg = "Some attribution sidecars could not be evaluated:\n- " + "\n- ".join(problems[:20]) \
              + ("\n... (more omitted)" if len(problems) > 20 else "")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {len(problems)} sidecar(s) skipped.\n{msg}")
    if not rows:
        raise ValueError(f"No usable attribution sidecars found in {attributions_dir}")

    df = pd.DataFrame(rows, columns=ERROR_COLUMNS).sort_values(["scm_id", "model", "explainer", "seed"])
    df = df.reset_index(drop=True)
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scms-dir", required=True)
    parser.add_argument("--attributions-dir", required=True)
    parser.add_argument("--output-csv", default="results/errors.csv")
    parser.add_argument("--strict", action="store_true", help="fail on any unusable sidecar")
    args = parser.parse_args()
    df = build_error_table(args.scms_dir, args.attributions_dir, args.output_csv, strict=args.strict)
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
