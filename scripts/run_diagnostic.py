"""Stage 5: train diagnostics, evaluate under grouped CV / LOGFO / designated split, write tables + figures.

Usage:
  python scripts/run_diagnostic.py --scms-dir data/scms --results-dir results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.diagnostic import run_diagnostic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scms-dir", default="data/scms")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--errors-csv", default=None)
    p.add_argument("--observables-csv", default=None)
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    res = Path(a.results_dir)
    run_diagnostic(a.errors_csv or res / "errors.csv", a.observables_csv or res / "observables.csv",
                   Path(a.scms_dir) / "manifest.csv", out_dir=res, targets=a.targets, n_boot=a.n_boot,
                   seed=a.seed, make_figures=not a.no_figures)


if __name__ == "__main__":
    main()
