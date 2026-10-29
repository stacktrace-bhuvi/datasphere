# Predicting Explanation Error Without Causal Ground Truth: A Held-Out-SCM Study of SHAP and LIME

**Course:** Advanced Data Analytics (Sem 5)  
**Institution:** PES University  
**Team:** DataSphere  

---

## Overview

Feature attribution methods such as SHAP and LIME are widely deployed to explain complex machine learning predictions. Practitioners frequently interpret these attributions causally—as indicators of which variables drive an outcome under intervention. However, in real-world deployments, causal ground truth is unavailable. When an explanation fails to reflect the underlying data-generating mechanism, this failure cannot be directly detected.

This project investigates whether **explanation error can be predicted using only quantities observable without causal ground truth**. Using Structural Causal Models (SCMs) as a controlled synthetic laboratory, we evaluate whether observable signals (such as feature dependence, multicollinearity, algorithmic stability, attribution sensitivity, concentration, and explainer disagreement) can serve as reliable diagnostics of explanation error, and whether such diagnostic models generalize across held-out SCM graph families.

---

## Motivation

When a feature attribution method assigns high importance to a variable, downstream decision-makers often assume that intervening on that variable will significantly alter the outcome. This assumption fails for multiple known reasons:
1. Feature attribution objectives typically target observational or conditional associations rather than interventional distributions.
2. Machine learning models can exploit spurious correlations or confounders that do not reflect true causal pathways.
3. Local attribution approximations introduce sampling and surrogate estimation errors.

Because true causal graphs and interventional outcomes are inaccessible in production settings, practitioners have no way to quantify explanation degradation. Rather than introducing new theoretical failure modes, this study treats explanation diagnostic evaluation as a meta-learning and prediction problem: **Can observable properties of the data, the model, and the explainer predict explanation error on unseen data-generating processes?**

---

## Method and Pipeline

The project uses a benchmark pipeline structured around synthetic SCMs with known causal graphs, mechanisms, and noise distributions. The evaluation pipeline consists of five stages:

```
[ SCM Generation ] 
        │  (Known structural equations, graphs, interventions)
        ▼
[ Ground-Truth Effect Computation ] 
        │  (Total interventional effect do(X_j), exact oracle SHAP of f*)
        ▼
[ Model Training & Explainer Execution ]
        │  (Train XGBoost, MLP; compute SHAP, LIME, Permutation Importance)
        ▼
[ Observable Metric Extraction & Error Decomposition ]
        │  (Extract dependence, VIF, sensitivity, disagreement; decompose error)
        ▼
[ Diagnostic Evaluation & Held-Out SCM Generalization ]
        │  (Train meta-models, benchmark against baselines, LOGFO evaluation)
```

1. **SCM Generation:** Generate synthetic SCMs across multiple graph families (e.g., Erdős–Rényi, scale-free, bipartite, tree-structured) varying structural mechanisms (linear, polynomial, non-linear splines), noise distributions, confounding, and mediation structures. The target outcome $Y$ is fixed as a sink node.
2. **Ground-Truth Derivation:** Compute the exact total interventional effect for each feature $X_j$ under the do-operator $\text{do}(X_j = x)$ via simulation and numerical integration, along with the exact oracle SHAP of the true generating function $f^*$.
3. **Model Fitting and Explanation:** Train predictive models (XGBoost and Multilayer Perceptrons) on observational draws from each SCM. Generate local and global feature attributions using SHAP (KernelSHAP, TreeSHAP), LIME, and permutation feature importance across multiple random seeds and bootstrap repeats.
4. **Observable Feature Extraction:** Compute diagnostic features that require no causal ground truth:
   - Data-level observables: Pairwise dependence, maximal correlation, Variance Inflation Factors (VIF), empirical sample size.
   - Model-level observables: Test error, local loss gradients, prediction variance, prediction concentration.
   - Explainer-level observables: Local explainer stability under perturbation, attribution sensitivity, pairwise explainer disagreement (e.g., rank divergence between SHAP and LIME).
5. **Diagnostic Meta-Learning:** Train diagnostic regression and classification models to predict attribution error from the observables. Evaluate transferability using stratified splits across SCM families and Leave-One-Graph-Family-Out (LOGFO) cross-validation. Compare against standard baselines (correlation-only, max VIF, model validation error, sample size).

---

## Error Decomposition

To pinpoint where explanations fail, we decompose the total error between an empirical explanation $\hat{E}(f, X_j)$ and the true causal estimand $\tau(X_j)$ into three orthogonal components:

$$\text{Total Error} = \underbrace{\left| \tau(X_j) - \phi_j(f^*) \right|}_{\text{Estimand Gap}} + \underbrace{\left| \phi_j(f^*) - \phi_j(f) \right|}_{\text{Model Misfit}} + \underbrace{\left| \phi_j(f) - \hat{E}(f, X_j) \right|}_{\text{Explainer Approximation Error}}$$

Where:
- $\tau(X_j)$ is the true total interventional effect under $\text{do}(X_j)$.
- $\phi_j(f^*)$ is the exact interventional SHAP attribution computed on the ground-truth function $f^*$.
- $\phi_j(f)$ is the exact SHAP attribution computed on the fitted model $f$.
- $\hat{E}(f, X_j)$ is the empirical attribution output by the chosen explainer (e.g., sample-based KernelSHAP or LIME surrogate).

### Component Definitions

1. **Estimand Gap:** Quantifies the inherent structural divergence between the game-theoretic attribution objective and the causal interventional estimand. This is structural and computed directly per SCM.
2. **Model Misfit:** Measures the discrepancy introduced when the predictive model $f$ fails to capture the true data-generating function $f^*$ (due to inductive bias, underfitting, or overfitting).
3. **Explainer Approximation Error:** Measures variance and bias introduced by finite-sample approximations, local linear surrogate modeling, or background dataset sampling in the explainer.

---

## Repository Structure

```
datasphere/
├── configs/                     # Experiment configurations (YAML)
│   ├── scm_generation.yaml      # Graph generator and structural mechanism parameters
│   ├── models.yaml              # Model architectures and hyperparameters (XGBoost, MLP)
│   └── explainers.yaml          # Explainer settings (sample counts, kernels, seeds)
├── src/
│   ├── scm/                     # Structural Causal Model generation and do-calculus
│   │   ├── generator.py         # Graph synthesis, structural equations, noise models
│   │   ├── interventional.py    # do(X_j) simulation and total effect calculation
│   │   └── oracle.py            # Ground truth f* evaluation and exact oracle SHAP
│   ├── models/                  # Predictive model training and evaluation
│   │   ├── xgboost_model.py     # XGBoost wrappers and hyperparameter tuning
│   │   └── mlp_model.py         # PyTorch/scikit-learn MLP implementations
│   ├── explainers/              # Explainer execution and attribution wrappers
│   │   ├── shap_wrapper.py      # KernelSHAP and TreeSHAP interfaces
│   │   ├── lime_wrapper.py      # LIME tabular explainer interface
│   │   └── permutation.py       # Permutation feature importance
│   ├── observables/             # Feature extraction without ground truth
│   │   ├── dependence.py        # Mutual information, correlation, VIF
│   │   ├── stability.py         # Explainer sensitivity and perturbation stability
│   │   └── disagreement.py      # Pairwise rank and magnitude disagreement metrics
│   ├── metrics/                 # Attribution evaluation and decomposition
│   │   ├── error_metrics.py     # Rank agreement, Normalized L2, Kendall tau
│   │   └── decomposition.py     # 3-way error decomposition implementation
│   ├── diagnostic/              # Meta-models predicting explanation error
│   │   ├── meta_regressor.py    # Error prediction models
│   │   ├── baselines.py         # Baseline diagnostic models (VIF, test error)
│   │   └── evaluation.py        # Grouped and Leave-One-Graph-Family-Out (LOGFO) evaluation
│   └── utils/                   # Manifest generation, I/O, and seed management
├── tests/                       # Unit and integration test suite
│   ├── test_scm.py
│   ├── test_models.py
│   ├── test_explainers.py
│   └── test_decomposition.py
├── results/                     # Structured outputs, logs, tables, and figures
│   ├── tables/
│   └── figures/
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.11 (64-bit)
- Git

### Installation

1. Clone the repository and checkout the working branch:
   ```bash
   git clone https://github.com/stacktrace-bhuvi/datasphere.git
   cd datasphere
   git checkout feat/scm-ground-truth
   ```

2. Create and activate a virtual environment:
   - On Linux/macOS:
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. Install required dependencies:
   ```bash
   python tasks.py install
   ```

### Task Runner and Pipeline Commands

The repository provides a cross-platform task runner [`tasks.py`](file:///c:/Users/Bhuvi/Desktop/sem5/ada/datasphere/tasks.py) (compatible with Windows PowerShell, Linux, and macOS without requiring `make`):

- **Install dependencies:**
  ```bash
  python tasks.py install
  ```

- **Run test suite:**
  ```bash
  python tasks.py test
  ```

- **Generate small sanity benchmark (10 SCMs):**
  ```bash
  python tasks.py generate-small
  ```

- **Generate full benchmark (100 SCMs):**
  ```bash
  python tasks.py generate-full
  ```

- **Run demonstrator script (Estimand Gap):**
  ```bash
  python -m src.scm.demo_estimand_gap
  ```

---

## Team and Work Split

The responsibilities below represent the initial module division for Team DataSphere. These assignments are collaborative suggestions and can be adjusted or swapped as development progresses.

| Member | GitHub | Responsibilities |
| :--- | :--- | :--- |
| **Bhuvi** | [@stacktrace-bhuvi](https://github.com/stacktrace-bhuvi) | SCM generation engine, graph family generators, `do()` simulation, ground-truth interventional effects, true function $f^*$, exact oracle SHAP, dataset manifest, and pipeline interface contracts. |
| **Mythri** | [@mythri2506a](https://github.com/mythri2506a) | Model training pipelines (XGBoost, MLP), attribution pipelines (SHAP, LIME, permutation importance), seed and bootstrap repeats, and attribution artifact persistence. |
| **Poojitha** | [@PoojithaBrpl](https://github.com/PoojithaBrpl) | Error metrics calculation (rank correlation, normalized $L_2$), 3-way error decomposition, end-to-end pipeline runner, results tables, and optional validation on semi-synthetic benchmarks (IHDP, Adult, California Housing). |
| **Diran** | [@diran07](https://github.com/diran07) | Observable feature extraction (VIF, dependence, sensitivity, disagreement), diagnostic meta-models, baseline diagnostics, grouped and Leave-One-Graph-Family-Out (LOGFO) evaluation, statistical significance tests, and figure generation. |

---

## Status

### Module Implementation Checklist

- [ ] **Structural Causal Models (`src/scm`)**
  - [ ] Graph family generators (Erdős–Rényi, scale-free, bipartite, tree)
  - [ ] Non-linear and linear mechanism assignment
  - [ ] Interventional `do(X_j)` numerical engine
  - [ ] Ground-truth function $f^*$ and oracle SHAP integration
  - [ ] Dataset manifest export and schema validation
- [ ] **Models and Explainers (`src/models`, `src/explainers`)**
  - [ ] XGBoost training and tuning pipeline
  - [ ] MLP training pipeline
  - [ ] SHAP interface (TreeSHAP and KernelSHAP)
  - [ ] LIME tabular explainer wrapper
  - [ ] Permutation feature importance integration
  - [ ] Bootstrap and random seed attribution replication
- [ ] **Observable Diagnostics (`src/observables`)**
  - [ ] Multicollinearity and dependence metrics (VIF, mutual info)
  - [ ] Explainer perturbation stability and sensitivity estimators
  - [ ] Multi-explainer rank and sign disagreement metrics
  - [ ] Model uncertainty and prediction concentration metrics
- [ ] **Metrics and Decomposition (`src/metrics`)**
  - [ ] Rank agreement and normalized $L_2$ error implementations
  - [ ] Three-way error decomposition module (Estimand gap, Model misfit, Approximation error)
  - [ ] Pipeline orchestrator and intermediate output schema checks
- [ ] **Diagnostic Evaluation (`src/diagnostic`)**
  - [ ] Meta-regressors and meta-classifiers for error prediction
  - [ ] Baseline diagnostic models (correlation-only, max VIF, test error)
  - [ ] Stratified SCM evaluation and Leave-One-Graph-Family-Out (LOGFO) cross-validation
  - [ ] Paired statistical hypothesis tests and bootstrap confidence intervals
- [ ] **Reporting and Benchmarks (`results`)**
  - [ ] Automated generation of comparison tables
  - [ ] Error distribution and LOGFO generalization plots
  - [ ] Optional semi-synthetic checks (IHDP, Adult Income, California Housing)

---

## References

1. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30.
2. **Janzing, D., Minorics, L., & Blöbaum, P. (2020).** Feature relevance quantification in explainable AI: A causality-based approach. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020)*, PMLR 108:1631-1640.
3. **Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Lakkaraju, H., & Zitnik, M. (2022).** OpenXAI: Towards a transparent evaluation of model explanations. *Advances in Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks*, 35.
