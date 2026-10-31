"""Generate compact reproducible summary tables from real errors.csv outputs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = ["scm_id", "model", "explainer", "seed", "total_error_l2",
            "total_error_rank", "estimand_gap", "model_misfit", "explainer_error"]


def summarize_errors(errors_csv: str | Path, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write grouped mean/std and explainer-comparison tables."""
    data = pd.read_csv(errors_csv)
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError(f"errors CSV missing required columns: {missing}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["total_error_l2", "total_error_rank", "estimand_gap",
               "model_misfit", "explainer_error"]

    grouped = data.groupby(["model", "explainer"], dropna=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(out / "error_summary_by_model_explainer.csv", index=False)

    # Pairedness is not assumed: this is descriptive aggregation only.
    comparison = data.groupby("explainer", dropna=False)[metrics].agg(["mean", "std", "count"])
    comparison.columns = [f"{metric}_{stat}" for metric, stat in comparison.columns]
    comparison = comparison.reset_index()
    comparison.to_csv(out / "error_summary_by_explainer.csv", index=False)

    return {"by_model_explainer": grouped, "by_explainer": comparison}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors-csv", default="results/errors.csv")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    tables = summarize_errors(args.errors_csv, args.output_dir)
    print(f"Wrote {len(tables['by_model_explainer'])} model/explainer groups and "
          f"{len(tables['by_explainer'])} explainer groups.")
