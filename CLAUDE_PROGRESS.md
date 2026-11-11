# CLAUDE_PROGRESS.md — DataSphere: Metrics & Results integration

> Durable progress log. "Completed" means executed and verified. "Attempted" means tried but not verified.
> Last updated: checkpoint 4 (code complete + tested + small E2E done; larger run + docs remain).

## Identity
- Project: "When Do Explanations Lie? Testing SHAP/LIME Against Causal Ground Truth" (team DataSphere, PES University ADA)
- Repo: https://github.com/stacktrace-bhuvi/datasphere
- User's role: Poojitha (@PoojithaBrpl) — **Metrics and Results**
- Working clone (sandbox): `/home/claude/work/repo_clone`; venv (sandbox, not in repo): `/home/claude/work/venv` (Python 3.11.18)
- Branch: `feat/poojitha-metrics-integration` (created from `main` @ `0011994`)
- Merged metrics work present: PR #3, commit `8525def` (verified in `git log`).

## Source-of-truth facts verified (checkpoint 1)
- The uploaded ZIP is **flat** (no dirs, no .git) and equals `origin/main` @ `0011994` modulo CRLF line endings (verified with `diff --strip-trailing-cr` on 22 files). The real clone from GitHub was used instead.
- Remote branch `origin/fix/link-attributions-to-metrics` exists but has **zero commits** beyond `main` (nothing pre-built).
- `requirements.txt` pins `xgboost>=2.0,<3.1` (comment: shap<=0.49 can't read xgboost>=3.1 models). User's local xgboost 3.2.0 is outside this pin -> `pip install -r requirements.txt` in a 3.11 venv.
- Confirmed integration gap: `src/explainers/io.py` sidecar has no `fitted_model_mean_abs_shap`; `src/metrics/pipeline_runner.py` requires it -> pipeline cannot run on real artifacts.
- README says "100 SCMs" in places; `tasks.py` + `configs/scm_default.yaml` (60 per family x 5 families) = **300 SCMs** is the real target.
- `scripts/make_mock_results.py` writes MOCK attributions/errors.csv into `results/` (same place as real output). Mock sidecars lack the reference field so the pipeline rejects them.

## Baseline test results (actually run)
- Env: Python 3.11.18, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.0.5, shap 0.51.0, lime 0.2.0.1, pyarrow 26.0.1, pytest 9.2.0
- Command: `cd repo_clone && ../venv/bin/python -m pytest tests -q -x -p no:cacheprovider`
- Result: **63 passed, 0 failed** (3 harmless shap PendingDeprecationWarnings), 23.7 s.
- (User's earlier "4 failures" were from shap/lime not installed.)

## Completed
- Repo audit; branch created; env built; baseline tests run; small benchmark generated:
  `python -m src.scm.generate --config configs/scm_small.yaml --output-dir data/scms_small --workers 2` -> 10 SCMs, 27 s (data/ is gitignored).

## Completed & verified (checkpoint 3)
- Reference SHAP integration (task 1): `src/explainers/reference.py` (new), `src/explainers/io.py`, `src/explainers/__init__.py`, `scripts/run_attributions.py` edited. Reference = exact enumeration SHAP of fitted model on GT explain/bg rows, computed once per scm x model x seed, `exact` explainer reuses it, legacy sidecars back-filled after model-identity check (exact artifact or models.csv test_mse).
  - `tests/test_reference.py`: 15 passed. Existing 63 tests still pass after changes (`pytest tests -q` -> 63 passed, 44 s).
- Pipeline hardening (task 2): `src/metrics/pipeline_runner.py` rewritten (PipelineInputError; validates filename/content, dims, finiteness, feature order, ref-vs-GT row provenance, sidecar-vs-parquet; strict default, `--allow-skipped` writes errors_skipped.csv; sorted + atomic output; permutation/global explainers -> NaN total_error_l2 & explainer_error). `tests/test_pipeline_runner.py`: 20 passed.
- REAL smoke (1 SCM `scm_0000_chain_mediation_d5`, seed 0, 2 models, 5 explainers): `python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results --limit 1 --seeds 0` -> 7.1 s, 17 attribution file pairs; pipeline -> 9 rows in results/errors.csv. exact/treeshap explainer_error=0.0000, kernelshap ~0.007, lime 0.11-0.20; estimand_gap 0.6690 == ground_truth.json estimand_gap.normalized_l2 (cross-check OK).

## Completed & verified (checkpoint 4)
- `src/metrics/results_tables.py` rewritten: SCM = unit of analysis, mean/SD/SEM/95% t-CI, n_scms/n_rows/<metric>_n; tables by_model_explainer, by_explainer, by_seed, by_family (needs --scms-dir manifest), components_by_model (explainer-independent terms de-duplicated). `tests/test_results_tables.py`: 9 passed. NOTE: std/CI are now across SCMs (old version: across rows) -- documented in module docstring.
- Metric audit: `error_metrics.py`/`decomposition.py` now reject non-1-D input (previously a (2,2) array silently matched a length-4 vector). Metric definitions UNCHANGED. `tests/test_error_metrics.py` covers ties, constant vectors, zero-norm (eps), NaN/inf, shape mismatch, telescoping-vs-non-additive norms.
- `tests/test_end_to_end.py` (7 tests): real SCM->ground truth->train->explain->errors->tables, legacy back-fill, mismatch refusal, skip-on-rerun.
- FULL SUITE: `../venv/bin/python -m pytest tests -q -p no:cacheprovider` -> **131 passed, 0 failed** (27 s).
- REAL SMALL BENCHMARK (all executed): `python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results --seeds 0 1 2` -> 10 SCMs x 3 seeds x {xgboost: exact,treeshap,kernelshap,lime,permutation; mlp: exact,kernelshap,lime,permutation} = **270 attribution pairs, 2m54s** (SCM0/seed0 was skipped = skip logic works). `python -m src.metrics.pipeline_runner --scms-dir data/scms_small --attributions-dir results/attributions --output-csv results/errors.csv` -> 270 rows (60 NaN in total_error_l2/explainer_error = exactly the permutation rows). `python -m src.metrics.results_tables --errors-csv results/errors.csv --output-dir results/tables --scms-dir data/scms_small` -> 5 tables.
- Observed (descriptive, n=10 SCMs, wide CIs): mean estimand_gap 0.568; mean model_misfit mlp 0.398 / xgboost 0.442; mean explainer_error: exact/treeshap 0.000, kernelshap 0.015-0.023, lime 0.154 (mlp) / 0.241 (xgboost); total_error_l2 mean ~0.57-0.70. Total error is dominated by estimand gap + model misfit, not explainer approximation (except LIME).
- Back-fill on real artifacts verified: stripped reference from 4 xgboost sidecars (scm_0000/seed0), pipeline raised actionable error, re-run restored reference bit-identically (only reference.compute_time_seconds differs), parquets untouched, errors.csv byte-identical.

## Remaining tasks (priority order)
1. Larger synthetic run (feasible subset; full 300 SCMs est. many hours on 2 cores) -> record exact counts.
2. README + docs/SCHEMAS.md updates (sidecar reference fields, errors.csv NaN semantics, tables, commands, runtime, limits); fix README checkboxes only where verified; tasks.py commands for run/errors/tables.
3. Optional datasets (IHDP/Adult/California Housing): NOT started; no compatible SCM ground truth -> document, do not force.
4. Final full test run; final commit(s) (local only; no push without approval).

## Current task / next action
Generate a medium benchmark: create `configs/scm_medium.yaml` (n_scms_per_family: 6 => 30 SCMs, n_samples 2000, d 5-10), generate into data/scms_medium, time it, then run run_attributions for seed 0 and pipeline+tables.

## Decisions that must not be lost
- Reference = exact enumeration SHAP (`src.scm.oracle.exact_shap`) of the FITTED model on the same explain/background rows as ground truth. Never copied from the empirical explainer, never `oracle_mean_abs_shap`.
- Component L2 norms are NOT additive; only the vector differences telescope.
- Permutation importance is global MSE-increase (not SHAP units): do not compute SHAP-unit L2 metrics for it.

## Git
Local commits on branch `feat/poojitha-metrics-integration` (see `git log`); NOTHING pushed.

## Uncommitted work (at checkpoint 4)
Modified: scripts/run_attributions.py, src/explainers/{__init__,io}.py, src/metrics/pipeline_runner.py. New: src/explainers/reference.py, tests/test_reference.py, tests/test_pipeline_runner.py, CLAUDE_*.md. No commits yet. Generated (gitignored) data: data/scms_small (10 SCMs), results/ (1-SCM smoke) -- valid, don't regenerate.
