# SCM Foundation Module: Interface Contract

This document specifies the exact API signatures, data contracts, and behavioral invariants for the SCM (Structural Causal Model) and Ground Truth module. Downstream modules (`src/models`, `src/explainers`, `src/metrics`, `src/diagnostic`) should interface directly with these classes and functions.

---

## 1. Core SCM Class (`src.scm.scm.SCM`)

```python
from src.scm.scm import SCM
```

### 1.1 Constructor & Builder

```python
@classmethod
def build(
    cls,
    family: str,
    d: int,
    seed: int,
    nonlinearity_prob: float = 0.5,
    noise_type: str = "gaussian",
    noise_scale: float = 1.0,
    interaction_prob: float = 0.0,
) -> SCM:
```
Procedurally instantiates a validated SCM from a specified graph family.

- **Parameters:**
  - `family`: One of `["chain_mediation", "confounded_fork", "feature_collider", "erdos_renyi_dag", "layered"]`.
  - `d`: Number of features ($5 \le d \le 10$).
  - `seed`: Integer RNG seed for reproducible graph structure and mechanisms.
  - `nonlinearity_prob`: Probability of assigning a non-linear mechanism (`tanh`, `sin`, `quadratic`, `relu`) to a node.
  - `noise_type`: `"gaussian"` or `"laplace"`.
  - `noise_scale`: Standard deviation / scale parameter $\sigma > 0$.
  - `interaction_prob`: Probability of pairwise feature interactions ($X_i \cdot X_j$).
- **Invariants Guaranteed:**
  - The graph is a directed acyclic graph (DAG).
  - Feature nodes are named `X1`, `X2`, ..., `Xd`.
  - Node `Y` is a strict sink (`out_degree("Y") == 0`).
  - All parents of `Y` are among `X1..Xd`.

---

### 1.2 `SCM.sample`

```python
def sample(self, n: int, seed: int) -> pd.DataFrame:
```
Draws observational samples from the joint distribution $P(X_1, \dots, X_d, Y)$.

- **Parameters:**
  - `n` (`int`): Number of observational instances to generate.
  - `seed` (`int`): Random seed ensuring deterministic draws.
- **Returns:**
  - `pd.DataFrame` with exact columns `['X1', 'X2', ..., 'Xd', 'Y']` of length $n$.
- **Behavior:**
  - Evaluates equations in topological order.
  - Uses Common Random Numbers (CRN) per node.

---

### 1.3 `SCM.sample_do`

```python
def sample_do(
    self,
    n: int,
    seed: int,
    interventions: dict[str, float | np.ndarray],
) -> pd.DataFrame:
```
Draws samples under interventional Pearl `do()` operations using Common Random Numbers (CRN).

- **Parameters:**
  - `n` (`int`): Number of instances.
  - `seed` (`int`): Random seed. Must match the seed used in `sample(n, seed)` when performing paired comparisons.
  - `interventions` (`dict[str, float | np.ndarray]`): Mapping from node names (e.g. `'X1'`) to fixed scalar values or arrays of length $n$.
- **Returns:**
  - `pd.DataFrame` with columns `['X1', ..., 'Xd', 'Y']`.
- **Invariants Guaranteed:**
  - Intervened nodes are fixed to the specified values.
  - Non-descendants of intervened nodes are **bit-for-bit identical** to `sample(n, seed)`.
  - Topological descendants are recomputed using the exact same exogenous noise draws as `sample(n, seed)`.

---

### 1.4 `SCM.true_effects`

```python
def true_effects(
    self,
    n_mc: int = 200_000,
    delta_sd: float = 1.0,
    seed: int = 0,
    signed: bool = False,
) -> np.ndarray:
```
Computes the exact total causal interventional effect $\tau_j$ for each feature $X_j$ on $Y$ under the do-operator.

- **Formula:**
  $$\tau_j = \mathbb{E}_{x \sim P(X_j)} \left[ \frac{\left| \mathbb{E}[Y \mid \text{do}(X_j = x + \delta \sigma_j)] - \mathbb{E}[Y \mid \text{do}(X_j = x - \delta \sigma_j)] \right|}{2} \right]$$
  If `signed=True`, the absolute value is omitted to preserve directionality.
- **Parameters:**
  - `n_mc` (`int`): Monte Carlo sample size (default `200_000`).
  - `delta_sd` (`float`): Perturbation factor in units of marginal standard deviation $\sigma_j$ (default `1.0`).
  - `seed` (`int`): Base seed for evaluation.
  - `signed` (`bool`): If `True`, returns signed effect; if `False`, returns mean absolute effect.
- **Returns:**
  - `np.ndarray` of shape `(d,)` containing $\tau_1, \dots, \tau_d$.

---

### 1.5 `SCM.f_star`

```python
def f_star(self, X: np.ndarray) -> np.ndarray:
```
Computes the true noiseless structural equation for outcome $Y$:
$$Y = f^*(X_1, \dots, X_d) = \mathbb{E}[Y \mid \text{parents}(Y)]$$

- **Parameters:**
  - `X` (`np.ndarray`): Array of shape `(n, d)` or `(d,)`. Column $j$ corresponds to feature $X_{j+1}$. Non-parents of $Y$ are ignored.
- **Returns:**
  - `np.ndarray` of shape `(n,)` (or scalar if 1D input).

---

### 1.6 Serialization

```python
def to_dict(self) -> dict[str, Any]:
def to_json(self, filepath: str | Path) -> None:

@classmethod
def from_dict(cls, data: dict[str, Any]) -> SCM:

@classmethod
def from_json(cls, filepath: str | Path) -> SCM:
```
Allows lossless round-trip serialization of the graph topology, structural equation parameters, noise models, and metadata.

---

## 2. Exact Shapley Oracle (`src.scm.oracle`)

```python
from src.scm.oracle import exact_shap, mean_abs_shap
```

### 2.1 `exact_shap`

```python
def exact_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    max_d: int = 10,
    chunk_size: int = 64,
) -> np.ndarray:
```
Computes exact interventional (marginal) Shapley values by full $2^d$ coalition enumeration.

- **Formula:**
  $$\phi_j(x) = \sum_{S \subseteq D \setminus \{j\}} \frac{|S|!(d - |S| - 1)!}{d!} \left( \mathbb{E}_{X_B}[f(x_{S \cup \{j\}}, (X_B)_{\overline{S \cup \{j\}}})] - \mathbb{E}_{X_B}[f(x_S, (X_B)_{\bar{S}})] \right)$$
- **Guaranteed Axioms:**
  - **Efficiency:** $\sum_{j=1}^d \phi_j(x) = f(x) - \mathbb{E}_{X_B}[f(X_B)]$ within $10^{-8}$.
  - **Linear consistency:** For $f(x) = \beta^T x + b$, $\phi_j(x) = \beta_j (x_j - \bar{x}_{B, j})$.
  - **Dummy player:** If feature $j$ is not in the active parents of $f$, $\phi_j(x) = 0$ for all $x$.
- **Parameters:**
  - `f`: Vectorized callable taking `(N, d)` and returning `(N,)`.
  - `X_explain`: `(n_explain, d)` array of instances to explain.
  - `X_background`: `(n_bg, d)` background/baseline distribution.
  - `max_d`: Safety check on maximum features ($d \le 10$).
  - `chunk_size`: Vectorization chunk size to bound memory consumption.
- **Returns:**
  - `phi`: `(n_explain, d)` array of Shapley values.

### 2.2 `mean_abs_shap`

```python
def mean_abs_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    max_d: int = 10,
    chunk_size: int = 64,
) -> np.ndarray:
```
Convenience function returning the mean absolute attribution across explained instances:
$$\bar{\phi}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x^{(i)})|$$
Returns array of shape `(d,)`.

---

## 3. Dataset Generator CLI (`src.scm.generate`)

### Command-line execution:
```bash
python -m src.scm.generate --config configs/scm_default.yaml --output-dir data/scms
```

### Outputs:
1. `data/scms/scms/<scm_id>.json`: Fully serializable SCM files.
2. `data/scms/manifest.csv`: Manifest index with the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `scm_id` | `str` | Unique SCM identifier (e.g. `scm_0001_chain_mediation_d6`) |
| `family` | `str` | Graph family (`chain_mediation`, `confounded_fork`, `feature_collider`, `erdos_renyi_dag`, `layered`) |
| `d` | `int` | Feature count ($5 \le d \le 10$) |
| `n_nonlinear` | `int` | Number of non-linear node mechanisms |
| `noise_type` | `str` | Exogenous noise family (`gaussian`, `laplace`) |
| `noise_scale` | `float` | Base noise scale $\sigma$ |
| `has_confounding` | `bool` | True if graph contains confounding structures |
| `mediation_share` | `float` | Fraction of features having indirect causal paths to $Y$ |
| `seed` | `int` | Random seed used for generation |
| `split` | `str` | `"train"` or `"heldout"` (grouped by graph family) |
