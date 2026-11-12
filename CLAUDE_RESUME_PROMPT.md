# Resume prompt (paste into a new Claude session)

You are continuing my DataSphere "Metrics and Results" task in the repository (https://github.com/stacktrace-bhuvi/datasphere, branch `feat/poojitha-metrics-integration`).

1. Read `CLAUDE_PROGRESS.md` first, completely.
2. Run `git status`, `git branch --show-current`, `git log --oneline -10`, and `git diff --stat` to see what actually changed; confirm which files/commands in the log really exist/ran.
3. Verify completed items by re-running only the cheap checks (`python -m pytest tests -q`), not expensive computations. Do NOT regenerate data under `data/` or `results/` if valid outputs already exist (check the log for paths and counts).
4. Continue from the first unfinished item in "Remaining tasks" / "Current task / next action".
5. Do not rewrite teammates' code beyond the minimal documented changes; do not push, merge, force-push or delete branches without my approval; never fabricate results.
6. Before stopping for any reason, update `CLAUDE_PROGRESS.md` (completed vs attempted, commands run, results, exact next action) and this file if the procedure changed.

Final state: work is complete, committed locally on `feat/poojitha-metrics-integration` (base 0011994) and exported as `datasphere_metrics_integration.patch`; the user is applying it to their Windows repo (see 'Windows PowerShell' in CLAUDE_PROGRESS.md). Do NOT start seeds 1-2 without the user's agreement (~66 min). Older status line follows.

Status at last checkpoint: all required work is done and committed locally except optional items (seeds 1-2 on the full benchmark, optional datasets); see 'Remaining tasks'. If nothing remains that the user wants, just verify `pytest tests -q` and report.

Environment: Python 3.11 venv with `pip install -r requirements.txt` (xgboost must satisfy `<3.1`).
