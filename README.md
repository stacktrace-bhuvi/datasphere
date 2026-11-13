# Predicting Explanation Error Without Causal Ground Truth: A Held-Out-SCM Study of SHAP and LIME

**Course:** Advanced Data Analytics (Sem 5)  
**Institution:** PES University  
**Team:** DataSphere (Section G) — Bhuvi S. Katti, Mythri S, Poojitha L. B. R., Diran Somanna

---

## Overview

Feature attribution methods such as SHAP and LIME are widely deployed to explain complex machine learning predictions. Practitioners frequently interpret these attributions causally—as indicators of which variables drive an outcome under intervention. However, in real-world deployments, causal ground truth is unavailable. When an explanation fails to reflect the underlying data-generating mechanism, this failure cannot be directly detected.

This project investigates whether **explanation error can be predicted using only quantities observable without causal ground truth**. Using Structural Causal Models (SCMs) as a controlled synthetic laboratory, we evaluate whether observable signals (feature dependence, multicollinearity, model fit, attribution stability and concentration, explainer disagreement) can serve as reliable diagnostics of explanation error, and whether such diagnostic models generalize across held-out SCM graph families.

---

## Motivation

When a feature attribution method assigns high importance to a variable, downstream decision-makers often assume that intervening on that variable will significantly alter the outcome. This assumption fails for multiple known reasons:
1. Feature attribution objectives typically target observational or conditional associations rather than interventional distributions.
2. Machine learning models can exploit spurious correlations or confounders that do not reflect true causal pathways.
3. Local attribution approximations introduce sampling and surrogate estimation errors.

Because true causal graphs and interventional outcomes are inaccessible in production settings, practitioners have no way to quantify explanation degradation. This study treats explanation diagnosis as a meta-learning and prediction problem: **Can observable properties of the data, the model, and the explainer predict explanation error on unseen data-generating processes?**

---

## Method and Pipeline

```
[ 1  SCM Generation ]                 src/scm/graph.py, mechanisms.py, scm.py, generate.py
        │  5 graph families, linear / non-linear mechanisms, do()-interventions
        ▼
[ 2  Ground-Truth Computation ]       src/scm/scm.py::true_effects, oracle.py, ground_truth.py
        │  total interventional effect tau_j, exact oracle SHAP of f*
        ▼
[ 3  Models + Explainers ]            src/models/train.py, src/explainers/attribution.py
        │  XGBoost, MLP; exact SHAP, TreeSHAP, KernelSHAP, LIME, permutation; 3 seeds
        ▼
[ 4  Error Decomposition + Observables ]   src/metrics/, src/observables/
        │  estimand gap / model misfit / explainer error;  VIF, MI, stability, disagreement, ...
        ▼
[ 5  Diagnostic + Held-Out Evaluation ]    src/diagnostic/
           meta-models vs baselines; grouped CV, leave-one-graph-family-out, paired tests
```

1. **SCM Generation.** 300 synthetic SCMs (60 per family) across five graph families — chain-mediation, confounded fork, feature collider, Erdős–Rényi DAG, layered — with 5–10 features, linear / `tanh` / `sin` / quadratic / ReLU / pairwise-interaction mechanisms, Gaussian noise, and `Y` a strict sink. 2000 observational draws per SCM, split 60/20/20.
2. **Ground truth.** Total interventional effect `tau_j = E_x |E[Y|do(X_j = x+sd)] - E[Y|do(X_j = x-sd)]| / 2` by nested Monte-Carlo with common random numbers, and exact interventional Shapley values of the true function `f*` on fixed explain (100 test rows) and background (50 train rows) sets.
3. **Models and explainers.** XGBoost and an MLP trained on the train split (early stopping on val). Five explainers on the same rows: brute-force exact SHAP of the fitted model (`exact`), TreeSHAP, KernelSHAP (budget `max(2d+2, 2^d/4)` coalitions, so it stays an approximation), LIME, permutation importance. Three seeds each.
4. **Error decomposition.** `tau -> Phi(f*) -> Phi(f) -> E_hat(f)` splits total error into *estimand gap* (what SHAP measures vs the causal effect), *model misfit* and *explainer approximation error*. Reported in raw magnitude space, on **importance shares** (scale-free, the headline metric) and in rank space (`1 - Kendall tau-b`).
5. **Observables (no ground truth).** Data level: |corr|, VIF, mutual information, feature–target association, `n_train`, `d`. Model level: test/val error, train–test gap, prediction spread, residual kurtosis, cross-model prediction agreement. Explainer level: attribution concentration (Gini, top-1 share, effective #features), sign consistency, seed stability, perturbation stability and sensitivity (Lipschitz estimate), disagreement between explainers and between models.
6. **Diagnostic meta-learning.** Gradient-boosted, random-forest and ridge meta-models predict explanation error from the observables and are compared with single-signal baselines (constant, correlation-only, max VIF, test error, sample size, disagreement-only, stability-only). Protocols: StratifiedGroupKFold over SCMs (in-distribution), **leave-one-graph-family-out (LOGFO)**, and the designated train-families → layered split. Paired cluster bootstrap over SCMs and Wilcoxon signed-rank tests quantify the gain over the best baseline.

---

## Headline results (full run: 300 SCMs × 2 models × 5 explainers × 3 seeds = 8100 explanations)

| | |
| :--- | :--- |
| Estimand gap (share $L_2$, perfect model + exact SHAP) | **0.31**; largest for chain-mediation (0.41), smallest for random DAGs (0.26) |
| Model misfit / explainer error (share $L_2$) | 0.09 (XGBoost), 0.14 (MLP) / KernelSHAP < 0.01, LIME 0.06, permutation 0.15–0.17 |
| Rank error vs $\tau$ (1 − Kendall $\tau_b$) | 0.43 for every explainer — better explainers do not change the causal ranking |
| Diagnostic, share-error target (GBM meta-model) | Spearman 0.65 grouped CV, **0.52 LOGFO** (AUROC 0.76), 0.48 held-out layered; best single-signal baseline 0.36–0.46 |
| Gain over best baseline (LOGFO) | $\Delta\rho$ = 0.17 [0.08, 0.25], Wilcoxon $p < 0.001$, better on 65 % of SCMs |
| Which component is predictable (LOGFO $\rho$) | explainer error **0.91**, model misfit **0.70**, estimand gap 0.34, rank error 0.22 |

The part of the error a practitioner can act on (model, explainer) is visible in ground-truth-free observables; the dominant structural part (SHAP answering a different question than $\text{do}(X_j)$) largely is not. Full tables: `results/tables/`, figures: `results/figures/`, report: `report/main.pdf`.

---

## Error Decomposition

To pinpoint where explanations fail, we decompose the total error between an empirical explanation $\hat{E}(f, X_j)$ and the true causal estimand $\tau(X_j)$ into three ordered components:

$$\text{Total Error} \le \underbrace{\left| \tau(X_j) - \phi_j(f^*) \right|}_{\text{Estimand Gap}} + \underbrace{\left| \phi_j(f^*) - \phi_j(f) \right|}_{\text{Model Misfit}} + \underbrace{\left| \phi_j(f) - \hat{E}(f, X_j) \right|}_{\text{Explainer Approximation Error}}$$

- $\tau(X_j)$: true total interventional effect under $\text{do}(X_j)$.
- $\phi_j(f^*)$: exact interventional SHAP on the ground-truth function $f^*$.
- $\phi_j(f)$: exact SHAP on the fitted model $f$ (the `exact` explainer).
- $\hat{E}(f, X_j)$: the empirical attribution (KernelSHAP, LIME, ...).

> **How the decomposition is computed in code.** All quantities are feature-wise *non-negative magnitude vectors* in order `X1..Xd`: `tau` → `oracle_mean_abs_shap` (SHAP of the true function $f^*$) → `fitted_model_mean_abs_shap` (exact SHAP of the fitted model $f$) → `mean_abs_attributions` (selected explainer). The three difference **vectors telescope exactly** to `tau − explainer`; their normalized **L2 norms do not add up** to the total (different denominators, triangle inequality), so the three columns of `errors.csv` must not be summed. See `docs/SCHEMAS.md` §3.
>
> Because $\tau$ (outcome units) and SHAP (contribution units) live on different scales, every component is also computed on importance shares $|v| / \|v\|_1$ (`*_share` columns) and in rank space (`*_rank`), which are the scale-free comparisons used in the report.

### Component Definitions

1. **Estimand Gap:** Quantifies the inherent structural divergence between the game-theoretic attribution objective and the causal interventional estimand. This is structural and computed directly per SCM.
2. **Model Misfit:** Measures the discrepancy introduced when the predictive model $f$ fails to capture the true data-generating function $f^*$ (due to inductive bias, underfitting, or overfitting).
3. **Explainer Approximation Error:** Measures variance and bias introduced by finite-sample approximations, local linear surrogate modeling, or background dataset sampling in the explainer.

---

## Repository Structure

```
datasphere/
├── configs/
│   ├── scm_default.yaml           # full benchmark: 300 SCMs, n=2000, outcome_noise_scale 0.4
│   └── scm_small.yaml             # 10-SCM sanity benchmark
├── docs/
│   ├── INTERFACE.md               # SCM / oracle API contract
│   └── SCHEMAS.md                 # file schemas for every stage
├── src/
│   ├── scm/                       # stage 1-2: graph.py, mechanisms.py, scm.py, oracle.py, ground_truth.py, generate.py
│   ├── models/train.py            # stage 3: XGBoost + MLP (fast batched predict for the oracle)
│   ├── explainers/                # stage 3: attribution.py (5 explainers), io.py (parquet + json sidecar)
│   ├── metrics/                   # stage 4a: error_metrics.py, decomposition.py, pipeline_runner.py, results_tables.py
│   ├── observables/               # stage 4b: dependence.py, model_level.py, stability.py, disagreement.py, extract.py
│   └── diagnostic/                # stage 5: dataset.py, meta_regressor.py, baselines.py, evaluation.py, stats.py, figures.py, report.py
├── scripts/
│   ├── run_attributions.py        # stage 3 runner (sharded)
│   ├── build_observables.py       # stage 4b runner (sharded)
│   ├── run_diagnostic.py          # stage 5 runner: tables + figures + diagnostic_summary.json
│   ├── make_figures.py            # regenerate figures from saved tables
│   └── make_mock_results.py       # schema-conformant mock outputs for tests
├── notebooks/DataSphere_pipeline.ipynb   # end-to-end walkthrough (deliverable; build_notebook.py regenerates it)
├── report/                        # IEEE-format report (LaTeX: main.tex + generated tables.tex + figures/)
├── deliverables/                  # one-pager, deck, video script, numbers.json and the scripts that build them
├── tests/                         # pytest suite (scm, oracle, models, explainers, metrics, observables, diagnostic, schemas)
├── requirements.txt
├── tasks.py                       # cross-platform task runner
└── README.md
```

Generated artefacts (ignored by git): `data/<run>/<scm_id>/{data.parquet, scm.json, meta.json, ground_truth.json}`, `data/<run>/manifest.csv`, `results/attributions/*.{parquet,json}`, `results/models.csv`, `results/errors.csv`, `results/observables.csv`, `results/tables/`, `results/figures/`, `results/diagnostic_summary.json`.

---

## Setup and Usage

### Prerequisites

- Python 3.11+ (64-bit)
- Git

### Installation

```bash
git clone https://github.com/stacktrace-bhuvi/datasphere.git
cd datasphere
python3 -m venv .venv && source .venv/bin/activate      # Windows: py -3 -m venv .venv ; .\.venv\Scripts\Activate.ps1
python tasks.py install
```

### Reproducing the study

```bash
python tasks.py test                 # ~1 min: unit + schema tests
python tasks.py generate-small       # 10 SCMs  (~1 min)
python tasks.py pipeline-small       # stages 3-5 on the small benchmark (~8 min, writes results_small/)

python tasks.py generate-full        # 300 SCMs (~45 min on 2 cores)
python tasks.py pipeline-full        # stages 3-5 on the full benchmark (~3 h on 2 cores)
```

`pipeline-*` runs, in order:

```bash
python scripts/run_attributions.py   --scms-dir data/scms --results-dir results --seeds 0 1 2   # [--shard k/n]
python scripts/build_observables.py  --scms-dir data/scms --results-dir results                  # [--shard k/n]
python -m src.metrics.pipeline_runner --scms-dir data/scms --attributions-dir results/attributions --output-csv results/errors.csv
python scripts/run_diagnostic.py     --scms-dir data/scms --results-dir results
python scripts/make_figures.py       --scms-dir data/scms --results-dir results    # figures only, no refit
```

Each stage is resumable: existing attribution files are skipped unless `--force` is given. The two sharded stages can be run as `--shard 0/2` and `--shard 1/2` in parallel; concatenate the `*.shardK.csv` outputs afterwards (`tasks.py merge-shards`).

Other commands: `python tasks.py dry-run` (clipping statistics without writing), `python tasks.py mock` (schema-conformant fake results), `python -m src.scm.demo_estimand_gap` (estimand-gap demonstrator).

---

## Metrics and Results Pipeline (attributions → errors → tables)

Prerequisite: `pip install -r requirements.txt` in a Python 3.11 virtual environment (`xgboost` must satisfy `>=2.0,<3.1`; a newer XGBoost breaks `shap` TreeExplainer on older SHAP releases).

```bash
# 1. SCMs + causal ground truth (data/ is git-ignored)
python tasks.py generate-small                 # 10 SCMs  -> data/scms_small
python tasks.py generate-full                  # 300 SCMs -> data/scms
# 2. Train models, run explainers, compute the fitted-model reference SHAP
python scripts/run_attributions.py --scms-dir data/scms_small --results-dir results --seeds 0 1 2
# 3. Error decomposition -> results/errors.csv
python -m src.metrics.pipeline_runner --scms-dir data/scms_small --attributions-dir results/attributions --output-csv results/errors.csv
# 4. Summary tables -> results/tables/*.csv
python -m src.metrics.results_tables --errors-csv results/errors.csv --output-dir results/tables --scms-dir data/scms_small
```

(`python tasks.py attributions|errors|tables [args]` wrap steps 2–4.) **Run the tests** with `python -m pytest tests -q` (or `python tasks.py test`); the suite includes a small real end-to-end run (SCM → ground truth → train → explain → errors → tables) and needs no pre-generated data. Step 2 is resumable: finished artifacts are skipped, `--force` recomputes. Every attribution sidecar stores `fitted_model_mean_abs_shap`, the **exact** SHAP of the fitted model on the same explain/background rows as the ground truth. It is computed once per SCM × model × seed (cost = 100 × 50 × 2^d model evaluations, exact only for d ≤ 10) and shared by all explainers; the `exact` explainer reuses it. Step 3 refuses artifacts without it (or with mismatching rows, dimensions, feature order, non-finite values) and lists the problems; `--allow-skipped` keeps the valid rows and writes `errors_skipped.csv`.

> `scripts/make_mock_results.py` writes **mock** files into `results/`. Do not mix them with real runs; use a different `--results-dir` for real experiments.

**Interpreting results.** `estimand_gap` = how far SHAP of the *true* function is from the causal effect; `model_misfit` = how far the *fitted* model's exact SHAP is from the true function's; `explainer_error` = how far the chosen explainer is from the fitted model's exact SHAP. A small `explainer_error` only means the explainer is faithful to the *model*; it says nothing about causal correctness (that is what `estimand_gap` and `model_misfit` measure). `permutation` importance is a global test-MSE increase, not a SHAP-unit quantity, so its L2-based columns are NaN. Tables are descriptive (unit = SCM, 95% t-intervals across SCMs), not hypothesis tests.

**Earlier verified run (Metrics and Results, before the `outcome_noise_scale` change, i.e. outcome noise 1.0; superseded by the headline results above).** Full synthetic benchmark: 300 SCMs (60 per family, d = 5…10) × {XGBoost: exact, TreeSHAP, KernelSHAP, LIME, permutation; MLP: exact, KernelSHAP, LIME, permutation} × **seed 0 only** = 2700 attribution artifacts → 2700 `errors.csv` rows (generation ≈ 20 min, attributions ≈ 33 min on 2 CPU cores). Seeds 1 and 2 were not run on the full benchmark (a 10-SCM run used 3 seeds). Mean estimand gap 0.483 (95% CI 0.468–0.498); mean model misfit 0.293 (XGBoost) / 0.317 (MLP); mean explainer error: exact/TreeSHAP 0.000, KernelSHAP 0.006–0.010, LIME 0.138 (MLP) / 0.185 (XGBoost). On this benchmark most of the error against causal ground truth comes from the estimand gap and model misfit, not from explainer approximation (except LIME). TreeSHAP agrees with the independent exact reference to < 1e-5 for every d. Optional validation on IHDP / Adult Income / California Housing was **not** done (they have no SCM ground truth compatible with the decomposition).

---

## Team and Work Split

| Member | GitHub | Responsibilities |
| :--- | :--- | :--- |
| **Bhuvi** | [@stacktrace-bhuvi](https://github.com/stacktrace-bhuvi) | SCM generation engine, graph family generators, `do()` simulation, ground-truth interventional effects, true function $f^*$, exact oracle SHAP, dataset manifest, and pipeline interface contracts. |
| **Mythri** | [@mythri2506a](https://github.com/mythri2506a) | Model training pipelines (XGBoost, MLP), attribution pipelines (SHAP, LIME, permutation importance), seed and bootstrap repeats, and attribution artifact persistence. |
| **Poojitha** | [@PoojithaBrpl](https://github.com/PoojithaBrpl) | Error metrics (rank correlation, normalized $L_2$, share-space metrics), 3-way error decomposition, end-to-end pipeline runner, results tables. |
| **Diran** | [@diran07](https://github.com/diran07) | Observable feature extraction (VIF, dependence, stability, disagreement), diagnostic meta-models, baseline diagnostics, grouped and leave-one-graph-family-out evaluation, statistical significance tests, and figure generation. |

---

## Status

- [x] **Structural Causal Models (`src/scm`)**
  - [x] Graph family generators (chain-mediation, confounded fork, feature collider, Erdős–Rényi, layered)
  - [x] Non-linear and linear mechanism assignment; `outcome_noise_scale` to control the SNR of `Y`
  - [x] Interventional `do(X_j)` numerical engine (nested CRN estimator)
  - [x] Ground-truth function $f^*$ and exact oracle SHAP
  - [x] Dataset manifest export and schema validation
- [x] **Models and Explainers (`src/models`, `src/explainers`)**
  - [x] XGBoost and MLP training pipelines (shared 60/20/20 split, early stopping on val)
  - [x] Exact SHAP of the fitted model, TreeSHAP, KernelSHAP, LIME, permutation importance
  - [x] Seed replication and attribution artifact persistence
- [x] **Metrics and Decomposition (`src/metrics`)**
  - [x] Rank (Kendall $\tau_b$), normalized $L_2$, share-space $L_2$, top-$k$ overlap
  - [x] Three-way decomposition (raw, share and rank space); fitted-model reference SHAP stored in every sidecar (`src/explainers/reference.py`)
  - [x] Error table builder and summary tables
- [x] **Observable Diagnostics (`src/observables`)**
  - [x] Dependence metrics (|corr|, VIF, mutual information, feature–target association)
  - [x] Model-level metrics (test/val error, over-fit gap, prediction spread, cross-model agreement)
  - [x] Attribution concentration, sign consistency, seed stability, perturbation stability, sensitivity
  - [x] Explainer–explainer and cross-model disagreement
- [x] **Diagnostic Evaluation (`src/diagnostic`)**
  - [x] Meta-regressors (GBM, RF, ridge) and single-signal baselines
  - [x] StratifiedGroupKFold, leave-one-graph-family-out and designated held-out evaluation
  - [x] Paired cluster bootstrap and Wilcoxon tests, bootstrap CIs, permutation feature importance
- [x] **Reporting (`results/`, `report/`, `notebooks/`)**
  - [x] Automated tables, figures and `diagnostic_summary.json`
  - [x] Notebook walkthrough and IEEE-format report
  - [ ] Optional semi-synthetic checks (IHDP, Adult Income, California Housing)

---

## References

1. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *NeurIPS 2017*, 30.
2. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*.
3. **Janzing, D., Minorics, L., & Blöbaum, P. (2020).** Feature relevance quantification in explainable AI: A causality-based approach. *AISTATS 2020*, PMLR 108:1631–1640.
4. **Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Lakkaraju, H., & Zitnik, M. (2022).** OpenXAI: Towards a transparent evaluation of model explanations. *NeurIPS 2022 Datasets and Benchmarks*.
5. **Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S., & Lakkaraju, H. (2022).** The disagreement problem in explainable machine learning: A practitioner's perspective. *arXiv:2202.01602*.
6. **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
