"""Cross-platform project task runner for DataSphere.

Usage:
    python tasks.py <command>

Available commands:
    install          Install project dependencies from requirements.txt
    test             Run pytest test suite
    generate-small   Generate a small benchmark of 10 SCMs (sanity check)
    generate-full    Generate the full benchmark of 300 SCMs from configs/scm_default.yaml
    mock             Generate mock results (attributions, errors, observables) conforming to SCHEMAS.md
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


COMMANDS = {
    "install": install,
    "test": test,
    "generate-small": generate_small,
    "generate-full": generate_full,
    "mock": mock,
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
