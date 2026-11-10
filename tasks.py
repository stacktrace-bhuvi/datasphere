"""Cross-platform project task runner for DataSphere.

Usage:
    python tasks.py <command>

Available commands:
    install          Install project dependencies from requirements.txt
    test             Run pytest test suite
    generate-small   Generate a small benchmark of 10 SCMs (sanity check)
    generate-full    Generate the full benchmark of 300 SCMs from configs/scm_default.yaml
    mock             Generate mock results (attributions, errors, observables) conforming to SCHEMAS.md
    dry-run          Run generator in dry-run mode on full config without disk writes
    pipeline-small   Stages 3-5 (attributions, observables, errors, diagnostic) on data/scms_small -> results_small/
    pipeline-full    Stages 3-5 on data/scms -> results/   (hours; stages are resumable)
    merge-shards     Concatenate results/*.shardK.csv written by sharded runs into models.csv / observables.csv
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def run_command(cmd: list[str]) -> int:
    """Run a subprocess command cross-platform without shell-specific syntax."""
    print(f"==> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return result.returncode


def install() -> int:
    """Install project dependencies from requirements.txt."""
    req_path = ROOT_DIR / "requirements.txt"
    if not req_path.exists():
        print(f"Error: requirements.txt not found at {req_path}", file=sys.stderr)
        return 1
    return run_command([sys.executable, "-m", "pip", "install", "-r", str(req_path)])


def test() -> int:
    """Run test suite using pytest."""
    tests_dir = ROOT_DIR / "tests"
    return run_command([sys.executable, "-m", "pytest", str(tests_dir), "-v"])


def generate_small() -> int:
    """Generate a small synthetic benchmark suite (10 SCMs)."""
    cfg = ROOT_DIR / "configs" / "scm_small.yaml"
    out_dir = ROOT_DIR / "data" / "scms_small"
    return run_command([
        sys.executable,
        "-m",
        "src.scm.generate",
        "--config",
        str(cfg),
        "--output-dir",
        str(out_dir),
    ])


def generate_full() -> int:
    """Generate the full synthetic benchmark suite (300 SCMs)."""
    cfg = ROOT_DIR / "configs" / "scm_default.yaml"
    out_dir = ROOT_DIR / "data" / "scms"
    return run_command([
        sys.executable,
        "-m",
        "src.scm.generate",
        "--config",
        str(cfg),
        "--output-dir",
        str(out_dir),
    ])


def mock() -> int:
    """Generate mock results (attributions, errors.csv, observables.csv) for small SCM set."""
    script_path = ROOT_DIR / "scripts" / "make_mock_results.py"
    return run_command([
        sys.executable,
        str(script_path),
        "--scms-dir",
        str(ROOT_DIR / "data" / "scms_small"),
        "--results-dir",
        str(ROOT_DIR / "results"),
    ])


def dry_run() -> int:
    """Run generator in dry-run mode on full config without disk writes."""
    cfg = ROOT_DIR / "configs" / "scm_default.yaml"
    return run_command([
        sys.executable,
        "-m",
        "src.scm.generate",
        "--config",
        str(cfg),
        "--dry-run",
        "--workers",
        "4",
    ])


def _pipeline(scms_dir: Path, results_dir: Path, seeds: list[str]) -> int:
    """Stages 3-5 in sequence; stops at the first failing stage."""
    steps = [
        [sys.executable, str(ROOT_DIR / "scripts" / "run_attributions.py"), "--scms-dir", str(scms_dir),
         "--results-dir", str(results_dir), "--seeds", *seeds],
        [sys.executable, str(ROOT_DIR / "scripts" / "build_observables.py"), "--scms-dir", str(scms_dir),
         "--results-dir", str(results_dir)],
        [sys.executable, "-m", "src.metrics.pipeline_runner", "--scms-dir", str(scms_dir),
         "--attributions-dir", str(results_dir / "attributions"), "--output-csv", str(results_dir / "errors.csv")],
        [sys.executable, str(ROOT_DIR / "scripts" / "run_diagnostic.py"), "--scms-dir", str(scms_dir),
         "--results-dir", str(results_dir)],
    ]
    for cmd in steps:
        code = run_command(cmd)
        if code != 0:
            return code
    return 0


def pipeline_small() -> int:
    """Run stages 3-5 on the small benchmark (2 seeds) into results_small/."""
    return _pipeline(ROOT_DIR / "data" / "scms_small", ROOT_DIR / "results_small", ["0", "1"])


def pipeline_full() -> int:
    """Run stages 3-5 on the full benchmark (3 seeds) into results/."""
    return _pipeline(ROOT_DIR / "data" / "scms", ROOT_DIR / "results", ["0", "1", "2"])


def merge_shards() -> int:
    """Concatenate results/models.shardK.csv and results/observables.shardK.csv."""
    import pandas as pd
    res = ROOT_DIR / "results"
    for stem in ("models", "observables"):
        parts = sorted(res.glob(f"{stem}.shard*.csv"))
        if not parts:
            print(f"no {stem}.shard*.csv in {res}")
            continue
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        keys = ["scm_id", "model", "seed"] + (["explainer"] if stem == "observables" else [])
        df = df.drop_duplicates(keys, keep="last").sort_values(keys)
        df.to_csv(res / f"{stem}.csv", index=False)
        print(f"wrote {res / (stem + '.csv')} ({len(df)} rows from {len(parts)} shards)")
    return 0


COMMANDS = {
    "install": install,
    "test": test,
    "generate-small": generate_small,
    "generate-full": generate_full,
    "mock": mock,
    "dry-run": dry_run,
    "pipeline-small": pipeline_small,
    "pipeline-full": pipeline_full,
    "merge-shards": merge_shards,
}


def print_help() -> None:
    print(__doc__)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    cmd_name = sys.argv[1].lower()
    if cmd_name not in COMMANDS:
        print(f"Unknown command: '{cmd_name}'\n", file=sys.stderr)
        print_help()
        sys.exit(1)

    exit_code = COMMANDS[cmd_name]()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
