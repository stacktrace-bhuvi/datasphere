"""Stage 4 orchestrator: build results/observables.csv from saved attributions.

One row per (scm_id, model, explainer, seed) that has an attribution artifact.
Everything is recomputed from data.parquet, the (deterministically re-trained)
model, and the saved attribution files. ground_truth.json is read ONLY for the
explain/background row indices, which define which rows were explained; tau and
the oracle are never touched.

Observable groups (see SCHEMAS.md section 4):
  data       dependence / multicollinearity / target association   (per scm)
  model      fit quality, over-fitting, cross-model agreement        (per scm, model, seed)
  explainer  concentration, seed stability, perturbation stability,
             sensitivity, explainer and cross-model disagreement     (per row)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.explainers import explain, supports
from src.explainers.io import artifact_prefix
from src.models import MODEL_NAMES, split_xy, train_model
from src.observables.dependence import data_observables
from src.observables.disagreement import cross_model_disagreement, explainer_disagreement
from src.observables.model_level import cross_model_agreement, model_observables
from src.observables.stability import (
    concentration_observables,
    perturb_inputs,
    perturbation_observables,
    seed_stability,
)

OBSERVABLE_KEYS = ["scm_id", "model", "explainer", "seed"]


def _available(attr_dir: Path, scm_id: str) -> list[tuple[str, str, int]]:
    """(model, explainer, seed) triples that have a parquet+json pair for this SCM."""
    out = []
    for p in sorted(attr_dir.glob(f"{scm_id}__*__*__seed*.json")):
        if not p.with_suffix(".parquet").exists():
            continue
        _, model, explainer, seed = p.stem.split("__")
        out.append((model, explainer, int(seed.replace("seed", ""))))
    return out


def _load_values(attr_dir: Path, scm_id: str, model: str, explainer: str, seed: int, feats: list[str]):
    prefix = artifact_prefix(scm_id, model, explainer, seed)
    df = pd.read_parquet(attr_dir / f"{prefix}.parquet")
    with open(attr_dir / f"{prefix}.json", encoding="utf-8") as f:
        side = json.load(f)
    return df[feats].to_numpy(np.float64), side


def observables_for_scm(scm_dir: Path, attr_dir: Path, sigma: float = 0.05,
                        perturb_rows_exact: int = 20, perturb_seed_offset: int = 7_919,
                        skip_perturbation: bool = False) -> list[dict]:
    scm_id = scm_dir.name
    df = pd.read_parquet(scm_dir / "data.parquet")
    with open(scm_dir / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    feats = [c for c in df.columns if c.startswith("X")]
    parts = split_xy(df, feats)
    X_exp = df.loc[gt["explain_row_indices"], feats].to_numpy(np.float64)
    X_bg = df.loc[gt["background_row_indices"], feats].to_numpy(np.float64)
    X_tr, X_te, y_te = parts["train"][0], parts["test"][0], parts["test"][1]

    triples = _available(attr_dir, scm_id)
    if not triples:
        return []
    data_obs = data_observables(parts["train"][0], parts["train"][1], seed=0)

    # Load every attribution once; group for seed-stability and disagreement lookups.
    values: dict[tuple[str, str, int], np.ndarray] = {}
    mean_abs: dict[tuple[str, str, int], np.ndarray] = {}
    for model, explainer, seed in triples:
        v, _ = _load_values(attr_dir, scm_id, model, explainer, seed, feats)
        values[(model, explainer, seed)] = v
        mean_abs[(model, explainer, seed)] = np.abs(v).mean(axis=0)

    # Re-train the models deterministically (same seed => same model) for model-level
    # observables and perturbation re-explanations.
    models: dict[tuple[str, int], object] = {}
    model_obs: dict[tuple[str, int], dict] = {}
    test_preds: dict[tuple[str, int], np.ndarray] = {}
    for model_name, seed in sorted({(m, s) for m, _, s in triples}):
        mdl = train_model(model_name, df, seed=seed)
        models[(model_name, seed)] = mdl
        model_obs[(model_name, seed)] = model_observables(mdl, parts)
        test_preds[(model_name, seed)] = mdl.predict(X_te)

    rows = []
    for model_name, explainer, seed in triples:
        key = (model_name, explainer, seed)
        row: dict = {"scm_id": scm_id, "model": model_name, "explainer": explainer, "seed": seed}
        row.update(data_obs)
        row.update(model_obs[(model_name, seed)])

        # cross-model agreement of predictions (other model family, same seed)
        others = [p for (m, s), p in test_preds.items() if s == seed and m != model_name]
        row["model_agreement"] = cross_model_agreement(test_preds[(model_name, seed)], others[0]) \
            if others else float("nan")

        # explainer-level: concentration, seed stability, disagreement
        row.update(concentration_observables(values[key]))
        row.update(seed_stability([v for (m, e, s), v in mean_abs.items()
                                   if m == model_name and e == explainer]))
        same_run = {e: v for (m, e, s), v in mean_abs.items() if m == model_name and s == seed}
        row.update(explainer_disagreement(explainer, same_run, reference="lime"))
        cross = [v for (m, e, s), v in mean_abs.items() if e == explainer and s == seed and m != model_name]
        row.update(cross_model_disagreement(mean_abs[key], cross[0]) if cross
                   else {"cross_model_disagreement_rank": float("nan"), "cross_model_disagreement_l2": float("nan")})

        # perturbation stability: re-run the explainer on slightly perturbed inputs
        if skip_perturbation:
            row.update({"perturb_stability_rank": float("nan"), "perturb_rel_change": float("nan"),
                        "sensitivity": float("nan"), "global_share_shift": float("nan")})
        else:
            mdl = models[(model_name, seed)]
            n_rows = perturb_rows_exact if explainer == "exact" else len(X_exp)
            pseed = seed + perturb_seed_offset
            if explainer == "permutation":
                # global explainer: perturb the rows it is computed on (the test split)
                X_te_p = perturb_inputs(X_te, X_tr, sigma, pseed)
                attr_p = explain(mdl, explainer, X_exp, X_bg, seed=seed, X_train=X_tr,
                                 X_test=X_te_p, y_test=y_te)
                row.update(perturbation_observables(values[key], attr_p.values, X_te, X_te_p))
            else:
                X_exp_p = perturb_inputs(X_exp[:n_rows], X_tr, sigma, pseed)
                attr_p = explain(mdl, explainer, X_exp_p, X_bg, seed=seed, X_train=X_tr,
                                 X_test=X_te, y_test=y_te)
                row.update(perturbation_observables(values[key][:n_rows], attr_p.values,
                                                    X_exp[:n_rows], X_exp_p))
        rows.append(row)
    return rows


def build_observables(scms_dir: str | Path, results_dir: str | Path, output_csv: str | Path | None = None,
                      scm_ids: Iterable[str] | None = None, sigma: float = 0.05,
                      skip_perturbation: bool = False, verbose: bool = True) -> pd.DataFrame:
    scms_dir, results_dir = Path(scms_dir), Path(results_dir)
    attr_dir = results_dir / "attributions"
    manifest = pd.read_csv(scms_dir / "manifest.csv")
    ids = list(scm_ids) if scm_ids is not None else manifest["scm_id"].tolist()
    rows = []
    for i, scm_id in enumerate(ids, 1):
        t0 = time.perf_counter()
        rows += observables_for_scm(scms_dir / scm_id, attr_dir, sigma=sigma,
                                    skip_perturbation=skip_perturbation)
        if verbose:
            print(f"[{i}/{len(ids)}] {scm_id}: {time.perf_counter() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df
