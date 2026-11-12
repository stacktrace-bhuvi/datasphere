# CLAUDE_PROGRESS.md — DataSphere: Metrics & Results integration

> Durable progress log. "Completed" means executed and verified. "Attempted" means tried but not verified.
> Last updated: checkpoint 7 (FINAL for this session: code, tests, docs, full 300-SCM seed-0 run all done; seeds 1-2 + optional datasets not done).

## Identity
- Project: "When Do Explanations Lie? Testing SHAP/LIME Against Causal Ground Truth" (team DataSphere, PES University ADA)
- Repo: https://github.com/stacktrace-bhuvi/datasphere
- User's role: Poojitha (@PoojithaBrpl) — **Metrics and Results**
- Work was done in a throw-away sandbox clone of GitHub `main` (Python 3.11.18 venv built from requirements.txt); paths of that sandbox are intentionally not recorded.
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
- Command: `python -m pytest tests -q -x` (in a Python 3.11 venv)
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
- FULL SUITE: `python -m pytest tests -q` -> **131 passed, 0 failed** (27 s).
- REAL SMALL BENCHMARK (all executed): `python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results --seeds 0 1 2` -> 10 SCMs x 3 seeds x {xgboost: exact,treeshap,kernelshap,lime,permutation; mlp: exact,kernelshap,lime,permutation} = **270 attribution pairs, 2m54s** (SCM0/seed0 was skipped = skip logic works). `python -m src.metrics.pipeline_runner --scms-dir data/scms_small --attributions-dir results/attributions --output-csv results/errors.csv` -> 270 rows (60 NaN in total_error_l2/explainer_error = exactly the permutation rows). `python -m src.metrics.results_tables --errors-csv results/errors.csv --output-dir results/tables --scms-dir data/scms_small` -> 5 tables.
- Observed (descriptive, n=10 SCMs, wide CIs): mean estimand_gap 0.568; mean model_misfit mlp 0.398 / xgboost 0.442; mean explainer_error: exact/treeshap 0.000, kernelshap 0.015-0.023, lime 0.154 (mlp) / 0.241 (xgboost); total_error_l2 mean ~0.57-0.70. Total error is dominated by estimand gap + model misfit, not explainer approximation (except LIME).
- Back-fill on real artifacts verified: stripped reference from 4 xgboost sidecars (scm_0000/seed0), pipeline raised actionable error, re-run restored reference bit-identically (only reference.compute_time_seconds differs), parquets untouched, errors.csv byte-identical.

## Final handoff state (checkpoint 8 — verified in session 3)
- Base commit: `001199469606d2bf4505911e9880e86826b02d29` (`0011994`, "Merge pull request #3 ..."), equals merge-base with origin/main.
- Branch `feat/poojitha-metrics-integration`; 4 commits on top of base (the 5th "final handoff" commit hash is in the patch/`git log`; see PATCH below):
  - `15a8523` feat(explainers): reference SHAP in sidecars
  - `672fabaf` feat(metrics): validated pipeline runner, SCM-level tables, metric tests
  - `58d7fde` docs: progress + resume files
  - `ac8a8cd` docs(metrics): README/SCHEMAS/tasks/config/.gitignore
  - plus one final commit "docs: final handoff fixes" (SCHEMAS 'orthogonal' wording, README test command, progress files). Nothing pushed.
- PATCH (source of truth): `datasphere_metrics_integration.patch`, produced by `git format-patch 0011994..HEAD --stdout`, delivered to the user as a chat attachment (file named exactly that). Contains only source/tests/docs/config (21 files, no data/results/secrets). Its SHA-256 is recorded in the final chat message.
- Verified in session 3 against real files (not memory): 4 commits + base; benchmark outputs on disk: 2700 sidecars, 2700 parquets, errors.csv 2700 rows, 300 SCMs, seed [0] only, 600 NaN total_error_l2 == 600 permutation rows, TreeSHAP max explainer_error 3.258e-07.
- Docs fixed in session 3: SCHEMAS.md "orthogonal" -> "ordered / not additive"; README now states the test command.

## Remaining tasks (priority order)
1. User: apply the patch to the Windows repo (steps below), run the tests, review, then push / open PR themselves.
2. (Optional, needs user agreement first) seeds 1 and 2 on the full benchmark: ~33 min per seed on 2 CPU cores (~66 min), resumable: `python scripts/run_attributions.py --scms-dir data/scms --results-dir results_full --seeds 1 2`, then re-run pipeline_runner + results_tables (idempotent, seconds).
3. (Optional) IHDP / Adult / California Housing: NOT started; no compatible causal ground truth for the SCM decomposition. No results exist or are claimed.
4. Observables (`results/observables.csv`) belong to a teammate; not implemented here.

## Windows PowerShell: apply the patch safely
```powershell
cd "C:\Users\POOJITHA LAKSHMI\OneDrive\Desktop\STUDY\ADA\Jackfruit\datasphere"
git status --short                                  # must be empty; otherwise commit/stash first
git fetch origin; git checkout main; git pull --ff-only origin main
git log --oneline -1                                # expect 0011994 (or later; if later, use --3way below)
git checkout -b feat/poojitha-metrics-integration
git apply --check "$HOME\Downloads\datasphere_metrics_integration.patch"   # dry run, changes nothing
git am --3way "$HOME\Downloads\datasphere_metrics_integration.patch"
git log --oneline -6; git diff --stat main..HEAD
# if anything goes wrong:  git am --abort   (restores the branch exactly)
```
## Install and run
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q                            # expect 131 passed
python tasks.py generate-small
python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results_small --seeds 0 1 2
python -m src.metrics.pipeline_runner --scms-dir data/scms_small --attributions-dir results_small/attributions --output-csv results_small/errors.csv
python -m src.metrics.results_tables --errors-csv results_small/errors.csv --output-dir results_small/tables --scms-dir data/scms_small
```

## Known limitations
- Exact reference needs d<=10 (2^d enumeration); larger d raises.
- permutation importance: L2-based columns NaN (not SHAP units).
- Descriptive stats only; explainers evaluated on different SCM/model subsets (TreeSHAP = XGBoost only); seed 0 only for the 300-SCM run.
- Local user env had xgboost 3.2.0 (outside requirements pin <3.1) -- install via requirements.txt in a 3.11 venv.
- The original ZIP is flat; work was done in a fresh clone of GitHub main (identical modulo CRLF). The user's local Windows repo must receive these changes via git (push/pull) -- nothing was copied back to it.

## Current task / next action
Session complete. Next action for the user: review `git log` / `git diff 0011994..HEAD`, push `feat/poojitha-metrics-integration` and open a PR. Optional: run seeds 1 2 (task 1 above).

## Decisions that must not be lost
- Reference = exact enumeration SHAP (`src.scm.oracle.exact_shap`) of the FITTED model on the same explain/background rows as ground truth. Never copied from the empirical explainer, never `oracle_mean_abs_shap`.
- Component L2 norms are NOT additive; only the vector differences telescope.
- Permutation importance is global MSE-increase (not SHAP units): do not compute SHAP-unit L2 metrics for it.

## Medium benchmark (executed, checkpoint 5)
- `configs/scm_medium.yaml` (new; = scm_default with 6 SCMs/family) -> `python -m src.scm.generate --config configs/scm_medium.yaml --output-dir data/scms_medium --workers 2` = 30 SCMs (d=5..10, 5 each; 6/family), 1m57s.
- `python scripts/run_attributions.py --scms-dir data/scms_medium --results-dir results_medium --seeds 0` = 270 pairs (30 SCM x 9 combos), ~3 min. Pipeline -> results_medium/errors.csv 270 rows (60 NaN = permutation). Tables in results_medium/tables.
- TreeSHAP explainer_error vs independent reference: max 0.00000 at every d=5..10 (strong validation of reference + pipeline).
- Descriptive (30 SCMs, seed 0): estimand_gap 0.511; model_misfit xgboost 0.299 / mlp 0.353; explainer_error kernelshap 0.006-0.011, lime 0.150 (mlp) / 0.199 (xgboost); total_error_l2 xgboost-exact 0.535 [0.487,0.583], mlp-exact 0.512 [0.468,0.556].
- Because it was fast, the FULL 300-SCM benchmark is being attempted (see Current task).

## Git
Branch `feat/poojitha-metrics-integration` (base `main` @ 0011994). Local commits only; NOTHING pushed. See `git log --oneline`.
