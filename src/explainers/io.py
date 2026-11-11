"""Read/write attribution artifacts exactly as specified in docs/SCHEMAS.md section 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.explainers.attribution import Attribution
from src.explainers.reference import REFERENCE_FIELD, ReferenceShap


def artifact_prefix(scm_id: str, model: str, explainer: str, seed: int) -> str:
    return f"{scm_id}__{model}__{explainer}__seed{seed}"


def _check_reference(reference: ReferenceShap, attr: Attribution, row_indices: list[int]) -> None:
    d = attr.values.shape[1]
    if reference.values.shape[1] != d or len(reference.feature_names) != d:
        raise ValueError(f"reference SHAP has {reference.values.shape[1]} features, attribution has {d}")
    if [int(i) for i in row_indices] != reference.explain_row_indices:
        raise ValueError("reference SHAP was computed on different explain rows than this attribution")


def save_attribution(out_dir: str | Path, scm_id: str, model_name: str, seed: int,
                     attr: Attribution, row_indices: list[int],
                     model_config: dict[str, Any] | None = None,
                     reference: ReferenceShap | None = None) -> Path:
    """Write <prefix>.parquet (per-row attributions) and <prefix>.json (sidecar).

    If `reference` (exact SHAP of the fitted model, see src/explainers/reference.py) is given,
    the sidecar also gets `fitted_model_mean_abs_shap` + a `reference` provenance block, which
    the metrics pipeline needs. Without it the sidecar is the legacy format (still valid, but
    the metrics pipeline will reject it with an actionable error).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = attr.values.shape[1]
    prefix = artifact_prefix(scm_id, model_name, attr.explainer, seed)

    df = pd.DataFrame(attr.values, columns=[f"X{j}" for j in range(1, d + 1)])
    df.insert(0, "row_index", np.asarray(row_indices, dtype=np.int64))
    df["base_value"] = attr.base_values
    df.to_parquet(out / f"{prefix}.parquet", index=False)

    sidecar = {
        "scm_id": scm_id,
        "model": model_name,
        "explainer": attr.explainer,
        "seed": int(seed),
        "n_explained_rows": int(len(df)),
        "mean_abs_attributions": attr.mean_abs.tolist(),
        "compute_time_seconds": round(float(attr.compute_time_seconds), 4),
        "is_global": bool(attr.is_global),
        "config": {"model": _jsonable(model_config or {}), "explainer": _jsonable(attr.config)},
    }
    if reference is not None:
        _check_reference(reference, attr, row_indices)
        sidecar.update(reference.sidecar_fields(shared_with_exact_explainer=attr.explainer == "exact"))
    _write_json_atomic(out / f"{prefix}.json", sidecar)
    return out / f"{prefix}.parquet"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def has_reference(sidecar_path: str | Path) -> bool:
    with open(sidecar_path, encoding="utf-8") as f:
        return REFERENCE_FIELD in json.load(f)


def add_reference_to_sidecar(sidecar_path: str | Path, reference: ReferenceShap) -> None:
    """Back-fill the reference fields into an existing legacy sidecar (explainer output untouched)."""
    path = Path(sidecar_path)
    with open(path, encoding="utf-8") as f:
        sidecar = json.load(f)
    d = len(sidecar["mean_abs_attributions"])
    if reference.values.shape[1] != d:
        raise ValueError(f"{path.name}: reference has {reference.values.shape[1]} features, sidecar has {d}")
    sidecar.update(reference.sidecar_fields(shared_with_exact_explainer=sidecar.get("explainer") == "exact"))
    _write_json_atomic(path, sidecar)


def load_attribution(out_dir: str | Path, scm_id: str, model: str, explainer: str,
                     seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = artifact_prefix(scm_id, model, explainer, seed)
    df = pd.read_parquet(Path(out_dir) / f"{prefix}.parquet")
    with open(Path(out_dir) / f"{prefix}.json", encoding="utf-8") as f:
        return df, json.load(f)


def _jsonable(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
