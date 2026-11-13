# DataSphere Benchmark and Results Schemas

This document defines the formal data contracts and file schemas for datasets, ground truth, model attributions, error decompositions, and observable metrics across Team DataSphere's evaluation pipeline.

---

## 1. SCM Dataset & Ground Truth (`data/<run>/<scm_id>/`)

Each generated SCM directory contains four artifacts:

```
data/<run>/<scm_id>/
├── data.parquet         # Observational draws with train/val/test splits
├── scm.json             # Serialized SCM definition (DAG, equations, mechanisms)
├── meta.json            # High-level dataset metadata
└── ground_truth.json    # Exact causal effects, oracle SHAP, and estimand gap
```

### 1.1 `data.parquet`

- **Row Count:** $n_{\text{samples}}$ (default 2000).
- **Columns:**
  - `X1` ... `Xd`: `float64` — Feature columns.
  - `Y`: `float64` — Outcome sink node.
  - `row_split`: `string` (or categorical) — Partition indicator strictly partitioned as 60% `"train"`, 20% `"val"`, and 20% `"test"`.

#### Worked Example Row:
| X1 | X2 | X3 | X4 | X5 | Y | row_split |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| -0.4128 | 1.1542 | -0.0931 | 0.8294 | -1.2405 | 2.1934 | `train` |

---

### 1.2 `scm.json`
Contains the complete DAG topology, noise models, mechanism types, coefficients, and standardization parameters required to re-instantiate the model via `SCM.from_json()`.

---

### 1.3 `meta.json`

JSON object containing SCM configuration:

```json
{
  "scm_id": "scm_0001_chain_mediation_d5",
  "family": "chain_mediation",
  "d": 5,
  "n_nonlinear": 2,
  "noise_type": "gaussian",
  "noise_scale": 1.0,
  "has_confounding": false,
  "mediation_share": 0.6,
  "standardize": true,
  "clip_bound": 10.0,
  "seed": 42,
  "split": "train",
  "n_samples": 2000
}
```

---

### 1.4 `ground_truth.json`

Contains ground-truth interventional effects, oracle Shapley values, and structural metrics:

```json
{
  "scm_id": "scm_0001_chain_mediation_d5",
  "d": 5,
  "n_repeats": 5,
  "tau": [0.8661, 0.6234, 0.4120, 0.1294, 0.0],
  "tau_se": [0.0082, 0.0064, 0.0051, 0.0031, 0.0],
  "signed_tau": [0.8661, 0.6234, 0.4120, 0.1294, 0.0],
  "signed_tau_se": [0.0082, 0.0064, 0.0051, 0.0031, 0.0],
  "explain_row_indices": [0, 4, 7, 12, ...],
  "background_row_indices": [1, 2, 3, 5, ...],
  "oracle_mean_abs_shap": [0.0, 0.0, 0.4120, 0.1294, 0.0],
  "k_oracle": 2,
  "estimand_gap": {
    "normalized_l1": 0.7321,
    "normalized_l2": 0.6845,
    "kendall_tau_b": 0.6325,
    "top_k_overlap": 0.5,
    "k": 2
  },
  "clipped_fractions": {
    "X1": 0.0,
    "X2": 0.0,
    "X3": 0.001,
    "X4": 0.0,
    "X5": 0.0,
    "Y": 0.002
  },
  "clipped_fraction_Y": 0.002
}
```

---

## 2. Model Attributions (`results/attributions/`)

Attributions are saved per experiment run in Parquet format accompanied by a JSON sidecar:

```
results/attributions/
├── {scm_id}__{model}__{explainer}__seed{k}.parquet
└── {scm_id}__{model}__{explainer}__seed{k}.json
```

Naming convention:
- `{scm_id}`: SCM identifier (e.g. `scm_0001_chain_mediation_d5`).
- `{model}`: Model family (`xgboost` or `mlp`).
- `{explainer}`: Explainer name (`exact`, `treeshap`, `kernelshap`, `lime`, `permutation`). `exact` is brute-force exact SHAP of the fitted model; `treeshap` is XGBoost-only; `permutation` is global (the same vector on every row, `is_global: true`).
- `seed{k}`: Random seed integer (e.g. `seed0`).

### 2.1 Parquet Schema

- **Row Count:** Number of explained test instances (fixed at 100).
- **Columns:**
  - `row_index`: `int64` — The original dataset index corresponding to the explained test instance.
  - `X1` ... `Xd`: `float64` — Local attribution for feature $X_j$.
  - `base_value`: `float64` — Explainer baseline / expected value $E[f(X)]$.

#### Worked Example Row:
| row_index | X1 | X2 | X3 | X4 | X5 | base_value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7 | 0.0124 | -0.3412 | 0.8190 | 0.1044 | -0.0021 | 0.4851 |

### 2.2 JSON Sidecar Schema

```json
{
  "scm_id": "scm_0001_chain_mediation_d5",
  "model": "xgboost",
  "explainer": "treeshap",
  "seed": 0,
  "n_explained_rows": 100,
  "mean_abs_attributions": [0.0125, 0.2841, 0.7932, 0.1104, 0.0035],
  "compute_time_seconds": 0.342,
  "is_global": false,
  "config": {
    "model": {"max_depth": 4, "n_estimators": 500},
    "explainer": {"feature_perturbation": "interventional"}
  },
  "fitted_model_mean_abs_shap": [0.0121, 0.2850, 0.7901, 0.1110, 0.0041],
  "reference": {
    "method": "exact_shap_enumeration",
    "feature_names": ["X1", "X2", "X3", "X4", "X5"],
    "n_explain": 100,
    "n_background": 50,
    "explain_row_indices": [0, 4, 7, "..."],
    "background_row_indices": [1, 2, 3, "..."],
    "compute_time_seconds": 0.09
  }
}
```

#### Reference SHAP fields (required by the metrics pipeline)

- `fitted_model_mean_abs_shap` (`float[d]`): mean absolute **exact** interventional SHAP of the **fitted** model $\Phi(f)$, in feature order `X1..Xd`. It is computed by coalition enumeration (`src.scm.oracle.exact_shap`) on exactly `explain_row_indices` / `background_row_indices` from `ground_truth.json`, **independently of the explainer** in the filename, and once per `scm × model × seed` (identical in every sidecar of that group). It is never the explainer's own vector and never `oracle_mean_abs_shap` (SHAP of the true SCM function $f^*$). For the `exact` explainer the two vectors are equal by construction (`reference.shared_with_exact_explainer: true`).
- `reference` (`object`): provenance — `method` (must be `exact_shap_enumeration`), `feature_names`, row counts, the explain/background row indices used (the pipeline checks they equal the ground-truth ones), and compute time. Exact enumeration needs $d \le 10$; larger $d$ raises an error instead of approximating.
- Sidecars written before this field existed are **rejected** by `src.metrics.pipeline_runner` with an actionable message. Re-running `python scripts/run_attributions.py` (without `--force`) back-fills them after verifying that the retrained model reproduces the old artifacts; `--force` recomputes everything.

---

## 3. Error Decomposition (`results/errors.csv`)

Built by `python -m src.metrics.pipeline_runner` from `ground_truth.json` and the attribution sidecars. One row per SCM, model, explainer and seed. The fitted-model term $\Phi(f)$ is read from `fitted_model_mean_abs_shap` in each sidecar (section 2).

### CSV Columns

| Column | Description |
| :--- | :--- |
| `scm_id`, `model`, `explainer`, `seed` | keys (`model` in `xgboost, mlp`; `explainer` in `exact, treeshap, kernelshap, lime, permutation`) |
| `total_error_l2` | $\|\tau - \hat{E}\|_2 / (\|\tau\|_2 + 10^{-12})$ |
| `total_error_rank` | $1 - \text{Kendall } \tau_b(\tau, \hat{E})$ |
| `estimand_gap` | $\|\tau - \Phi(f^*)\|_2 / (\|\tau\|_2 + \epsilon)$ |
| `model_misfit` | $\|\Phi(f^*) - \Phi(f)\|_2 / (\|\Phi(f^*)\|_2 + \epsilon)$ |
| `explainer_error` | $\|\Phi(f) - \hat{E}\|_2 / (\|\Phi(f)\|_2 + \epsilon)$ |
| `total_error_share_l2`, `estimand_gap_share`, `model_misfit_share`, `explainer_error_share` | the same four distances computed between **importance shares** $s(v) = |v| / \|v\|_1$ (scale-free; the headline magnitude metric) |
| `estimand_gap_rank`, `model_misfit_rank`, `explainer_error_rank` | $1 - \text{Kendall } \tau_b$ between consecutive stages of the decomposition |
| `top_k_overlap` | fraction of the top-$k$ causal features (k = #features with $\tau_j > 0$) recovered in the top-$k$ of $\hat{E}$ |

Component norms are **not additive**: the feature-wise vectors telescope, their norms do not. For the global `permutation` explainer (MSE units) `total_error_l2` and `explainer_error` are NaN; the share and rank columns are still reported.

**Notes.**
- All vectors are non-negative magnitudes: `tau` and `oracle_mean_abs_shap` from `ground_truth.json`, `fitted_model_mean_abs_shap` and `mean_abs_attributions` from the sidecar. $\Phi(f^*)$ is the mean-abs oracle SHAP, $\Phi(f)$ the mean-abs exact SHAP of the fitted model.
- The component *vectors* ($\tau-\Phi(f^*)$, $\Phi(f^*)-\Phi(f)$, $\Phi(f)-\hat E$) telescope exactly to $\tau-\hat E$. Their normalized L2 *norms* use different denominators and generally do **not** add up to `total_error_l2`; do not sum the three columns.
- `estimand_gap` and `model_misfit` do not depend on the explainer (identical across explainers of the same scm/model/seed); only `explainer_error` and the total do.
- **Global explainers** (`permutation`, `is_global: true`) report an increase in test MSE, not SHAP-unit magnitudes. For them `total_error_l2` and `explainer_error` are written as **empty/NaN**; `total_error_rank`, `estimand_gap` and `model_misfit` are still reported.
- Constant vectors have undefined Kendall tau-b; it is reported as 0.0 (so `total_error_rank = 1.0`). A zero-norm denominator is guarded by the $10^{-12}$ term.
- Rows are sorted by `scm_id, model, explainer, seed`; the file is rebuilt from the artifacts on every run (never appended).

#### Worked Example Row:
```csv
scm_id,model,explainer,seed,total_error_l2,total_error_rank,estimand_gap,model_misfit,explainer_error,total_error_share_l2,estimand_gap_share,model_misfit_share,explainer_error_share,estimand_gap_rank,model_misfit_rank,explainer_error_rank,top_k_overlap
scm_0001_chain_mediation_d6,xgboost,kernelshap,0,0.5260,1.0,0.6690,0.2681,0.0070,0.3262,0.7011,0.3789,0.0067,0.3675,0.3675,0.0,1.0
```

---

## 4. Observable Features (`results/observables.csv`)

Built by `scripts/build_observables.py`. Every column is computable **without causal ground truth**: from the training split, the (deterministically re-trained) model, held-out labels, and the saved attributions. `ground_truth.json` is only read for the explain/background row indices. One row per SCM, model, explainer and seed.

### CSV Columns

| Group | Column | Description |
| :--- | :--- | :--- |
| keys | `scm_id`, `model`, `explainer`, `seed` | |
| data | `max_abs_corr`, `mean_abs_corr`, `frac_corr_gt_0_3` | pairwise \|Pearson r\| between features (train split) |
| data | `max_vif`, `mean_vif` | variance inflation factors |
| data | `mutual_info_max`, `mutual_info_mean` | pairwise mutual information (k-NN estimator, nats) |
| data | `max_abs_corr_xy`, `mean_abs_corr_xy`, `mi_xy_max`, `mi_xy_mean` | feature–target association |
| data | `n_train`, `d` | training rows, number of features |
| model | `test_mse`, `test_r2`, `val_mse`, `train_r2`, `overfit_gap` | fit quality; `overfit_gap = train_r2 - test_r2` |
| model | `pred_var_ratio` | Var(predictions on test) / Var(y test) |
| model | `resid_kurtosis` | excess kurtosis of test residuals |
| model | `model_agreement` | Pearson r between XGBoost and MLP predictions on test rows (same seed) |
| explainer | `attribution_gini`, `top1_share`, `effective_features` | concentration of mean-\|attribution\| shares |
| explainer | `sign_consistency` | share-weighted fraction of rows whose local sign matches the feature's majority sign |
| explainer | `local_dispersion` | mean over features of std(local attr) / mean\|attr\| |
| explainer | `seed_stability_rank`, `seed_stability_l2` | mean pairwise Kendall $\tau_b$ / share-$L_2$ between seeds of the same explainer (NaN if one seed) |
| explainer | `disagreement_rank_mean`, `disagreement_l2_mean`, `disagreement_topk_mean` | mean disagreement with the other explainers run on the same model and seed |
| explainer | `disagreement_rank_vs_lime`, `disagreement_l2_vs_lime` | disagreement with LIME as a fixed reference (LIME itself vs the mean of the others) |
| explainer | `cross_model_disagreement_rank`, `cross_model_disagreement_l2` | same explainer and seed, XGBoost vs MLP |
| explainer | `perturb_stability_rank`, `perturb_rel_change`, `global_share_shift` | attributions re-computed after Gaussian input perturbation ($\sigma = 0.05$ feature SDs); `exact` uses the first 20 explain rows |
| explainer | `sensitivity` | $\max_i \|\Delta \text{attr}_i\| / \|\Delta x_i\|$ (Lipschitz estimate) |

`src.diagnostic.dataset.observable_columns()` selects the diagnostic inputs automatically as every numeric column that is not a key, a target, or a structural column from the manifest (`family`, `split`, `has_confounding`, `mediation_share`, ...). One-hot indicators of `model` and `explainer` are appended.

---

## 5. Diagnostic Outputs (`results/tables/`, `results/figures/`, `results/diagnostic_summary.json`)

| File | Content |
| :--- | :--- |
| `tables/diagnostic_scores.csv` | `target, protocol, method, fold, r2, spearman, mae, rmse, n, auroc_high_error`. `fold` is `pooled`, a CV fold index, a held-out family, or `by_method:<model>/<explainer>` |
| `tables/diagnostic_predictions.csv` | out-of-sample prediction for every row under every method and protocol |
| `tables/paired_tests.csv` | meta-model vs each baseline: paired cluster bootstrap over SCMs (MAE and Spearman gain with 95% CI and p) and Wilcoxon signed-rank p |
| `tables/metric_bootstrap_ci.csv` | bootstrap CI of pooled Spearman and MAE per method and protocol |
| `tables/feature_importance_<target>.csv` | permutation importance of observables (trained on train families, scored on the held-out family) |
| `tables/error_summary_*.csv`, `estimand_gap_by_family.csv`, `error_by_family_explainer.csv` | descriptive tables |
| `diagnostic_summary.json` | headline numbers used in the report |

---

## 6. Summary Tables (`results/tables/`)

Produced by `python -m src.metrics.results_tables` from `errors.csv` (descriptive statistics only). The SCM is the unit of analysis: rows of the same SCM (seeds, pooled models) are averaged first, then mean / `std` / `sem` / 95% t-interval (`ci95_low`, `ci95_high`) are computed across SCMs. Columns per metric: `<metric>_mean, _std, _sem, _ci95_low, _ci95_high, _n` (`_n` = SCMs with a finite value); per group: `n_scms`, `n_rows`.

| File | Group keys |
| :--- | :--- |
| `error_summary_by_model_explainer.csv` | `model, explainer` |
| `error_summary_by_explainer.csv` | `explainer` (models pooled; TreeSHAP exists only for XGBoost) |
| `error_summary_by_seed.csv` | `model, explainer, seed` |
| `error_summary_by_family.csv` | `family, model, explainer` (needs `--scms-dir` with `manifest.csv`) |
| `error_summary_components_by_model.csv` | `model` — `estimand_gap`, `model_misfit` counted once per scm/model/seed |

These are not paired tests and make no causal claim.
