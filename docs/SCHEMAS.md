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
- `{explainer}`: Explainer name (`treeshap`, `kernelshap`, `lime`, `permutation`).
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
  "config": {
    "max_depth": 4,
    "n_estimators": 100
  }
}
```

---

## 3. Error Decomposition (`results/errors.csv`)

Tracks total explanation error and the three orthogonal decomposition components per SCM, model, and explainer.

### CSV Columns:
1. `scm_id` (`string`): SCM identifier.
2. `model` (`string`): `xgboost` or `mlp`.
3. `explainer` (`string`): `treeshap`, `kernelshap`, `lime`, or `permutation`.
4. `seed` (`int`): Model/explainer training seed.
5. `total_error_l2` (`float`): Normalized L2 error between attribution vector and true total causal effect $\tau$:
   $$\frac{\|\tau - \hat{E}\|_2}{\|\tau\|_2 + 10^{-12}}$$
6. `total_error_rank` (`float`): $1.0 - \text{Kendall\_tau\_b}(\tau, \hat{E})$.
7. `estimand_gap` (`float`): Inherent gap between exact oracle SHAP on $f^*$ and $\tau$:
   $$\frac{\|\tau - \Phi(f^*)\|_2}{\|\tau\|_2 + 10^{-12}}$$
8. `model_misfit` (`float`): Error due to model prediction error on $f^*$:
   $$\frac{\|\Phi(f^*) - \Phi(f)\|_2}{\|\Phi(f^*)\|_2 + 10^{-12}}$$
9. `explainer_error` (`float`): Approximation error of the explainer algorithm vs exact attribution on fitted model $f$:
   $$\frac{\|\Phi(f) - \hat{E}\|_2}{\|\Phi(f)\|_2 + 10^{-12}}$$

#### Worked Example Row:
```csv
scm_id,model,explainer,seed,total_error_l2,total_error_rank,estimand_gap,model_misfit,explainer_error
scm_0001_chain_mediation_d5,xgboost,treeshap,0,0.6912,0.3675,0.6845,0.0842,0.0121
```

---

## 4. Observable Features (`results/observables.csv`)

Contains all diagnostic metrics observable **without causal ground truth**, computed solely from observational training data, fitted model behavior, and explainer perturbations.

### CSV Columns:
1. `scm_id` (`string`): SCM identifier.
2. `model` (`string`): Model name (`xgboost` or `mlp`).
3. `explainer` (`string`): Explainer name.
4. `seed` (`int`): Random seed.
5. `max_vif` (`float`): Maximum Variance Inflation Factor across all features.
6. `mean_vif` (`float`): Average Variance Inflation Factor across features.
7. `mutual_info_max` (`float`): Maximum pairwise mutual information between features.
8. `mutual_info_mean` (`float`): Average pairwise mutual information among features.
9. `test_mse` (`float`): Mean squared error of model $f$ on test split.
10. `test_r2` (`float`): Coefficient of determination $R^2$ on test split.
11. `explainer_stability` (`float`): Attribution stability under Gaussian input perturbation $\sigma_{\text{perturb}}=0.05$.
12. `explainer_sensitivity` (`float`): Maximum attribution gradient / Lipshitz continuity estimator.
13. `prediction_concentration` (`float`): Gini coefficient of feature attributions across instances.
14. `explainer_disagreement_rank` (`float`): Kendall rank distance between target explainer and baseline LIME.
15. `explainer_disagreement_l2` (`float`): Normalized L2 distance between target explainer and baseline LIME.
16. `n_train` (`int`): Number of training samples ($1200$).

#### Worked Example Row:
```csv
scm_id,model,explainer,seed,max_vif,mean_vif,mutual_info_max,mutual_info_mean,test_mse,test_r2,explainer_stability,explainer_sensitivity,prediction_concentration,explainer_disagreement_rank,explainer_disagreement_l2,n_train
scm_0001_chain_mediation_d5,xgboost,treeshap,0,3.412,1.824,0.482,0.194,0.241,0.865,0.941,0.182,0.345,0.185,0.224,1200
```
