"""Stage 4: compute ground-truth-free observables for every saved attribution.

Writes results/observables.csv (one row per scm x model x explainer x seed).

Usage:
  python scripts/build_observables.py --scms-dir data/scms --results-dir results
  python scripts/build_observables.py --shard 0/2   # first half of the SCMs (run 2 processes)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from src.observables import build_observables  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scms-dir", default="data/scms")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-csv", default=None, help="default: <results-dir>/observables[.shardK].csv")
    p.add_argument("--sigma", type=float, default=0.05, help="perturbation size in feature SDs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="k/n: process every n-th SCM starting at k")
    p.add_argument("--skip-perturbation", action="store_true")
    a = p.parse_args()

    manifest = pd.read_csv(Path(a.scms_dir) / "manifest.csv")
    ids = manifest["scm_id"].tolist()[: a.limit]
    suffix = ""
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        ids = ids[k::n]
        suffix = f".shard{k}"
    out = a.output_csv or str(Path(a.results_dir) / f"observables{suffix}.csv")
    df = build_observables(a.scms_dir, a.results_dir, out, scm_ids=ids, sigma=a.sigma,
                           skip_perturbation=a.skip_perturbation)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {out}")


if __name__ == "__main__":
    main()
