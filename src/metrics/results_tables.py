"""Summary tables from `results/errors.csv` (descriptive statistics only).

Unit of analysis = the SCM. Rows for the same SCM share one ground truth and one dataset, so they are
not independent. Within each group, repeated rows of an SCM (seeds, and models when pooled) are first
averaged to one value per SCM; mean / SD / SE / 95% t-interval are then taken ACROSS SCMs.
`n_scms` is the number of SCMs in the group, `n_rows` the number of error rows behind them and
`<metric>_n` the number of SCMs with a finite value for that metric (permutation importance has NaN
`total_error_l2`/`explainer_error`, see pipeline_runner).

These are descriptive summaries of explanation error on this synthetic benchmark. Explainers are not
evaluated on identical SCM subsets (TreeSHAP is XGBoost-only) and no paired test is applied, so
differences between rows are not tests of causal or statistical significance.

Tables written to <output_dir>:
  error_summary_by_model_explainer.csv   [model, explainer]
  error_summary_by_explainer.csv         [explainer]  (models pooled)
  error_summary_by_seed.csv              [model, explainer, seed]
  error_summary_by_family.csv            [family, model, explainer]  (needs --scms-dir manifest)
  error_summary_components_by_model.csv  [model]: estimand_gap, model_misfit (explainer-independent,
                                         de-duplicated over explainers before averaging)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ["scm_id", "model", "explainer", "seed", "total_error_l2",
            "total_error_rank", "estimand_gap", "model_misfit", "explainer_error"]
METRICS = ["total_error_l2", "total_error_rank", "estimand_gap", "model_misfit", "explainer_error"]
KEY = ["scm_id", "model", "explainer", "seed"]


def load_errors(errors_csv: str | Path) -> pd.DataFrame:
    """Read and validate errors.csv (raises ValueError with a specific reason)."""
    path = Path(errors_csv)
    if not path.exists():
        raise ValueError(f"errors CSV not found: {path} (run python -m src.metrics.pipeline_runner first)")
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"errors CSV is empty: {path}") from e
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError(f"errors CSV missing required columns: {missing}")
    if data.empty:
        raise ValueError(f"errors CSV has no rows: {path}")
    dup = data.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"errors CSV has duplicate (scm_id, model, explainer, seed) rows, e.g. "
                         f"{data.loc[dup, KEY].head(3).values.tolist()}")
    for c in METRICS:
        data[c] = pd.to_numeric(data[c], errors="raise")
        if np.isinf(data[c]).any():
            raise ValueError(f"column {c} contains infinite values")
    return data


def _aggregate(data: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    per_scm = data.groupby(group + ["scm_id"], dropna=False)
    unit = per_scm[metrics].mean()                       # NaN only if every row of the unit is NaN
    unit["n_rows"] = per_scm.size()
    unit = unit.reset_index()
    out = []
    for keys, g in unit.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, keys))
        row["n_scms"], row["n_rows"] = int(len(g)), int(g["n_rows"].sum())
        for m in metrics:
            v = g[m].dropna().to_numpy(float)
            n = v.size
            mean = float(v.mean()) if n else np.nan
            sd = float(v.std(ddof=1)) if n > 1 else np.nan
            sem = sd / np.sqrt(n) if n > 1 else np.nan
            half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
            row.update({f"{m}_mean": mean, f"{m}_std": sd, f"{m}_sem": sem,
                        f"{m}_ci95_low": mean - half if n > 1 else np.nan,
                        f"{m}_ci95_high": mean + half if n > 1 else np.nan, f"{m}_n": n})
        out.append(row)
    return pd.DataFrame(out)


def _add_family(data: pd.DataFrame, scms_dir: str | Path) -> pd.DataFrame:
    manifest = Path(scms_dir) / "manifest.csv"
    if not manifest.exists():
        raise ValueError(f"manifest not found: {manifest}")
    fam = pd.read_csv(manifest)[["scm_id", "family"]]
    merged = data.merge(fam, on="scm_id", how="left")
    unknown = merged.loc[merged["family"].isna(), "scm_id"].unique().tolist()
    if unknown:
        raise ValueError(f"SCMs absent from {manifest}: {unknown[:5]}")
    return merged


def summarize_errors(errors_csv: str | Path, output_dir: str | Path,
                     scms_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Write the summary tables (see module docstring) and return them by name."""
    data = load_errors(errors_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "by_model_explainer": _aggregate(data, ["model", "explainer"], METRICS),
        "by_explainer": _aggregate(data, ["explainer"], METRICS),
        "by_seed": _aggregate(data, ["model", "explainer", "seed"], METRICS),
        # estimand_gap/model_misfit do not depend on the explainer: count each (scm, model, seed) once.
        "components_by_model": _aggregate(
            data.drop_duplicates(["scm_id", "model", "seed"]), ["model"], ["estimand_gap", "model_misfit"]),
    }
    if scms_dir is not None:
        tables["by_family"] = _aggregate(_add_family(data, scms_dir), ["family", "model", "explainer"], METRICS)

    files = {"by_model_explainer": "error_summary_by_model_explainer.csv",
             "by_explainer": "error_summary_by_explainer.csv",
             "by_seed": "error_summary_by_seed.csv",
             "by_family": "error_summary_by_family.csv",
             "components_by_model": "error_summary_components_by_model.csv"}
    for name, df in tables.items():
        df.to_csv(out / files[name], index=False)
    return tables


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--errors-csv", default="results/errors.csv")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--scms-dir", default=None, help="SCM folder with manifest.csv (adds the by-family table)")
    args = parser.parse_args()
    tables = summarize_errors(args.errors_csv, args.output_dir, args.scms_dir)
    print("Wrote: " + ", ".join(f"{k} ({len(v)} groups)" for k, v in tables.items()))
