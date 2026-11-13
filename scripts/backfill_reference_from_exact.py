"""Back-fill `fitted_model_mean_abs_shap` + `reference` into legacy sidecars from the saved `exact` artifact.

The `exact` explainer and the fitted-model reference are the same algorithm
(`src.scm.oracle.exact_shap` on the fitted model, same explain/background rows), see
`attribution_from_reference` in src/explainers/reference.py. For runs made before the reference
integration, this script rebuilds the reference from each (scm, model, seed)'s `exact` parquet
instead of retraining every model and redoing 2^d enumeration (which is what
`scripts/run_attributions.py` does when it back-fills).

Checks before writing: the exact parquet's row_index equals the ground-truth explain rows, its
mean |SHAP| equals the exact sidecar, and the exact sidecar has the same d as each target sidecar.
Sidecars that already carry a reference are left untouched.

    python scripts/backfill_reference_from_exact.py --scms-dir data/scms --attributions-dir results/attributions
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.explainers.io import add_reference_to_sidecar, has_reference  # noqa: E402
from src.explainers.reference import ReferenceShap, feature_names_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scms-dir", default="data/scms")
    ap.add_argument("--attributions-dir", default="results/attributions")
    a = ap.parse_args()
    scms, attr = Path(a.scms_dir), Path(a.attributions_dir)

    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for p in sorted(attr.glob("*.json")):
        scm_id, model, _, seed = p.stem.split("__")
        groups[(scm_id, model, int(seed.replace("seed", "")))].append(p)

    filled = skipped = 0
    problems = []
    gt_cache: dict[str, dict] = {}
    for (scm_id, model, seed), paths in groups.items():
        todo = [p for p in paths if not has_reference(p)]
        if not todo:
            skipped += len(paths)
            continue
        exact_json = attr / f"{scm_id}__{model}__exact__seed{seed}.json"
        exact_pq = exact_json.with_suffix(".parquet")
        if not exact_pq.exists():
            problems.append(f"{scm_id}/{model}/seed{seed}: no exact artifact; use run_attributions.py to back-fill")
            continue
        if scm_id not in gt_cache:
            gt_cache[scm_id] = json.load(open(scms / scm_id / "ground_truth.json", encoding="utf-8"))
        gt = gt_cache[scm_id]
        side = json.load(open(exact_json, encoding="utf-8"))
        df = pd.read_parquet(exact_pq)
        d = len(side["mean_abs_attributions"])
        cols = feature_names_for(d)
        if df["row_index"].tolist() != gt["explain_row_indices"]:
            problems.append(f"{exact_pq.name}: rows differ from ground-truth explain rows")
            continue
        vals = df[cols].to_numpy(float)
        if not np.allclose(np.abs(vals).mean(0), side["mean_abs_attributions"], rtol=1e-6, atol=1e-12):
            problems.append(f"{exact_json.name}: sidecar disagrees with its parquet")
            continue
        ref = ReferenceShap(values=vals, base_values=df["base_value"].to_numpy(float), feature_names=cols,
                            explain_row_indices=[int(i) for i in gt["explain_row_indices"]],
                            background_row_indices=[int(i) for i in gt["background_row_indices"]],
                            compute_time_seconds=float(side.get("compute_time_seconds", 0.0)))
        for p in todo:
            add_reference_to_sidecar(p, ref)
            filled += 1
    print(f"back-filled {filled} sidecars, {skipped} already had a reference, {len(problems)} problems")
    for m in problems[:20]:
        print("  -", m)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
