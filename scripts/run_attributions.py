"""Train models and run explainers on every SCM in a benchmark folder.

Writes:
  results/attributions/<scm_id>__<model>__<explainer>__seed<k>.{parquet,json}
  results/models.csv   (one row per SCM x model x seed: test_mse, test_r2, ...)

Usage:
  python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results
  python scripts/run_attributions.py --models xgboost --explainers exact treeshap --seeds 0

Every sidecar also stores `fitted_model_mean_abs_shap` (exact SHAP of the fitted model, computed
once per scm x model x seed and shared by all explainers) -- required by src.metrics.pipeline_runner.
Existing legacy sidecars without it are back-filled (after a model-identity check) unless --force.

  python scripts/run_attributions.py --shard 0/2   # half the SCMs; run --shard 1/2 in parallel
  (shards write models.shardK.csv; `python tasks.py merge-shards` concatenates them)
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.explainers import (  # noqa: E402
    EXPLAINER_NAMES, add_reference_to_sidecar, artifact_prefix, attribution_from_reference,
    compute_reference_shap, explain, has_reference, save_attribution, supports,
)
from src.models import MODEL_NAMES, split_xy, train_model  # noqa: E402


class _LazyReference:
    """Exact SHAP of one fitted model, computed on first use and shared by all explainers."""

    def __init__(self, model, X_exp, X_bg, explain_idx, bg_idx):
        self._args = (model, X_exp, X_bg, explain_idx, bg_idx)
        self._ref = None

    def get(self):
        if self._ref is None:
            self._ref = compute_reference_shap(*self._args)
        return self._ref


def _verify_same_model(attr_dir: Path, scm_id: str, model_name: str, seed: int, model, ref) -> None:
    """Before back-filling a reference into OLD sidecars, check the retrained model is the one
    that produced them. Uses the saved `exact` artifact if present, else models.csv test_mse."""
    exact_json = attr_dir / f"{artifact_prefix(scm_id, model_name, 'exact', seed)}.json"
    if exact_json.exists():
        with open(exact_json, encoding="utf-8") as f:
            old = np.asarray(json.load(f)["mean_abs_attributions"], dtype=float)
        if not np.allclose(old, ref.mean_abs, rtol=1e-5, atol=1e-8):
            raise RuntimeError(
                f"{scm_id}/{model_name}/seed{seed}: retrained model differs from the one behind the "
                "existing attribution files (library version or hardware drift?). "
                "Back-filling would pair mismatched artifacts; re-run with --force instead.")
        return
    models_csv = attr_dir.parent / "models.csv"
    if models_csv.exists():
        m = pd.read_csv(models_csv)
        hit = m[(m.scm_id == scm_id) & (m.model == model_name) & (m.seed == seed)]
        if len(hit) and not np.isclose(float(hit.iloc[-1]["test_mse"]), model.metrics["test_mse"],
                                       rtol=1e-6, atol=0):
            raise RuntimeError(
                f"{scm_id}/{model_name}/seed{seed}: retrained test_mse differs from models.csv; "
                "re-run with --force instead of back-filling.")
        return
    print(f"  WARNING {scm_id}/{model_name}/seed{seed}: cannot verify model identity for back-fill "
          "(no exact artifact or models.csv row)", flush=True)


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
            ref = _LazyReference(model, X_exp, X_bg, explain_idx, bg_idx)
            for ex_name in explainers:
                if not supports(model, ex_name):
                    continue
                prefix = artifact_prefix(scm_id, model_name, ex_name, seed)
                target, sidecar = attr_dir / f"{prefix}.parquet", attr_dir / f"{prefix}.json"
                if target.exists() and sidecar.exists() and not force:
                    if not has_reference(sidecar):      # legacy artifact: add reference only
                        r = ref.get()
                        _verify_same_model(attr_dir, scm_id, model_name, seed, model, r)
                        add_reference_to_sidecar(sidecar, r)
                    continue
                if ex_name == "exact":                  # exact explainer == the reference algorithm
                    attr = attribution_from_reference(ref.get())
                else:
                    attr = explain(model, ex_name, X_exp, X_bg, seed=seed,
                                   X_train=parts["train"][0], X_test=parts["test"][0],
                                   y_test=parts["test"][1])
                save_attribution(attr_dir, scm_id, model_name, seed, attr, explain_idx,
                                 model_config=model.config, reference=ref.get())
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scms-dir", default="data/scms_small")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--explainers", nargs="+", default=list(EXPLAINER_NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=None, help="only the first N SCMs (quick checks)")
    p.add_argument("--shard", default=None, help="k/n: process every n-th SCM starting at k (parallel runs)")
    p.add_argument("--force", action="store_true", help="recompute existing attribution files")
    a = p.parse_args()

    scms_dir, res_dir = Path(a.scms_dir), Path(a.results_dir)
    manifest = pd.read_csv(scms_dir / "manifest.csv")
    scm_ids = manifest["scm_id"].tolist()[: a.limit]
    suffix = ""
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        scm_ids, suffix = scm_ids[k::n], f".shard{k}"
    attr_dir = res_dir / "attributions"
    res_dir.mkdir(parents=True, exist_ok=True)
    models_csv = res_dir / f"models{suffix}.csv"
    rows = pd.read_csv(models_csv).to_dict("records") if models_csv.exists() else []

    for i, scm_id in enumerate(scm_ids, 1):
        t0 = time.perf_counter()
        new_rows = run_scm(scms_dir / scm_id, attr_dir, a.models, a.explainers, a.seeds, a.force)
        keys = {(r["model"], int(r["seed"])) for r in new_rows}
        rows = [r for r in rows if not (r["scm_id"] == scm_id and (r["model"], int(r["seed"])) in keys)] + new_rows
        # written after every SCM so an interrupted run loses nothing
        pd.DataFrame(rows).sort_values(["scm_id", "model", "seed"]).to_csv(models_csv, index=False)
        print(f"[{i}/{len(scm_ids)}] {scm_id} done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"Model metrics written to {models_csv}")


if __name__ == "__main__":
    main()
