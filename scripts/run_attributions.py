"""Train models and run explainers on every SCM in a benchmark folder.

Writes:
  results/attributions/<scm_id>__<model>__<explainer>__seed<k>.{parquet,json}
  results/models.csv   (one row per SCM x model x seed: test_mse, test_r2, ...)

Usage:
  python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results
  python scripts/run_attributions.py --models xgboost --explainers exact treeshap --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from src.explainers import EXPLAINER_NAMES, explain, save_attribution, supports  # noqa: E402
from src.models import MODEL_NAMES, split_xy, train_model  # noqa: E402


def run_scm(scm_dir: Path, attr_dir: Path, models, explainers, seeds, force=False) -> list[dict]:
    scm_id = scm_dir.name
    df = pd.read_parquet(scm_dir / "data.parquet")
    with open(scm_dir / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    feats = [c for c in df.columns if c.startswith("X")]
    explain_idx, bg_idx = gt["explain_row_indices"], gt["background_row_indices"]
    X_exp = df.loc[explain_idx, feats].to_numpy(float)
    X_bg = df.loc[bg_idx, feats].to_numpy(float)
    parts = split_xy(df, feats)

    rows = []
    for model_name in models:
        for seed in seeds:
            model = train_model(model_name, df, seed=seed)
            rows.append({"scm_id": scm_id, "model": model_name, "seed": seed, **model.metrics})
            for ex_name in explainers:
                if not supports(model, ex_name):
                    continue
                target = attr_dir / f"{scm_id}__{model_name}__{ex_name}__seed{seed}.parquet"
                if target.exists() and not force:
                    continue
                attr = explain(model, ex_name, X_exp, X_bg, seed=seed,
                               X_train=parts["train"][0], X_test=parts["test"][0],
                               y_test=parts["test"][1])
                save_attribution(attr_dir, scm_id, model_name, seed, attr, explain_idx,
                                 model_config=model.config)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scms-dir", default="data/scms_small")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--explainers", nargs="+", default=list(EXPLAINER_NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=None, help="only the first N SCMs (quick checks)")
    p.add_argument("--force", action="store_true", help="recompute existing attribution files")
    a = p.parse_args()

    scms_dir, res_dir = Path(a.scms_dir), Path(a.results_dir)
    manifest = pd.read_csv(scms_dir / "manifest.csv")
    scm_ids = manifest["scm_id"].tolist()[: a.limit]
    attr_dir = res_dir / "attributions"

    all_rows = []
    for i, scm_id in enumerate(scm_ids, 1):
        t0 = time.perf_counter()
        all_rows += run_scm(scms_dir / scm_id, attr_dir, a.models, a.explainers, a.seeds, a.force)
        print(f"[{i}/{len(scm_ids)}] {scm_id} done in {time.perf_counter() - t0:.1f}s", flush=True)

    res_dir.mkdir(parents=True, exist_ok=True)
    models_csv = res_dir / "models.csv"
    new = pd.DataFrame(all_rows)
    if models_csv.exists():
        old = pd.read_csv(models_csv)
        new = pd.concat([old, new]).drop_duplicates(["scm_id", "model", "seed"], keep="last")
    new.sort_values(["scm_id", "model", "seed"]).to_csv(models_csv, index=False)
    print(f"Model metrics written to {models_csv}")


if __name__ == "__main__":
    main()
