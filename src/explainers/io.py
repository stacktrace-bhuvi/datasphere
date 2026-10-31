"""Read/write attribution artifacts exactly as specified in docs/SCHEMAS.md section 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.explainers.attribution import Attribution


def artifact_prefix(scm_id: str, model: str, explainer: str, seed: int) -> str:
    return f"{scm_id}__{model}__{explainer}__seed{seed}"


def save_attribution(out_dir: str | Path, scm_id: str, model_name: str, seed: int,
                     attr: Attribution, row_indices: list[int],
                     model_config: dict[str, Any] | None = None) -> Path:
    """Write <prefix>.parquet (per-row attributions) and <prefix>.json (sidecar)."""
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
    with open(out / f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    return out / f"{prefix}.parquet"


def load_attribution(out_dir: str | Path, scm_id: str, model: str, explainer: str,
                     seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = artifact_prefix(scm_id, model, explainer, seed)
    df = pd.read_parquet(Path(out_dir) / f"{prefix}.parquet")
    with open(Path(out_dir) / f"{prefix}.json", encoding="utf-8") as f:
        return df, json.load(f)


def _jsonable(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
