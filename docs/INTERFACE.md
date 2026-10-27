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
    standardize: bool = True,
    clip_bound: float | None = 10.0,
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
  - `standardize` (`bool`): Default `True`. Incoming parent linear combinations are scaled by $1 / \sqrt{k}$ (where $k$ is the number of parents) and non-linear activations (e.g. `quadratic`, `relu`) are centered and scaled. This strictly prevents exponential variance compounding and explosion across deep graphs.
  - `clip_bound` (`float | None`): Default `10.0`. Bounds the structural equation output within $[-B, B]$ to guarantee numerical stability.
- **Invariants Guaranteed:**
  - The graph is a directed acyclic graph (DAG).
  - Feature nodes are named `X1`, `X2`, ..., `Xd`.
  - Node `Y` is a strict sink (`out_degree("Y") == 0`).
  - All parents of `Y` are among `X1..Xd`.

---

### 1.2 `SCM.sample`

```python
def sample(self, n: int, seed: int, return_noise: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
```
Draws observational samples from the joint distribution $P(X_1, \dots, X_d, Y)$.

- **Parameters:**
  - `n` (`int`): Number of observational instances to generate.
  - `seed` (`int`): Random seed ensuring deterministic draws.
  - `return_noise` (`bool`): If `True`, returns `(df, noise_dict)` for residual verification.
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
    n_ref: int = 50,
    n_noise_per_ref: int = 5_000,
    mode: str = "auto",
) -> np.ndarray:
```
Computes the exact total causal interventional effect $\tau_j$ for each feature $X_j$ on $Y$ under the do-operator.

- **Formula:**
  $$\tau_j = \mathbb{E}_{x \sim P(X_j)} \left[ \frac{\left| \mathbb{E}[Y \mid \text{do}(X_j = x + \delta \sigma_j)] - \mathbb{E}[Y \mid \text{do}(X_j = x - \delta \sigma_j)] \right|}{2} \right]$$
  If `signed=True`, the absolute value is omitted to preserve directionality.
- **Evaluation Modes:**
  1. **`mode="auto"` (Default):**
     - If the SCM is purely linear (`self.is_linear_scm`), uses single-pass CRN ($n_{\text{mc}} = 200,000$), where noise strictly cancels out in the difference and the result is exact in $\approx 1.5$ seconds for $d=10$.
     - If the SCM contains non-linear or interaction mechanisms, automatically invokes the **nested estimator** (`n_ref=50, n_noise_per_ref=5000`) to take the absolute value AFTER averaging over noise, preventing Jensen's inequality upward bias.
  2. **`mode="nested"`:**
     - Explicitly evaluates $n_{\text{ref}}$ reference values with $n_{\text{noise\_per\_ref}}$ noise draws per reference value using tensor broadcasting. Runtime for $d=10$ is $\approx 3.0$ seconds.
  3. **`mode="single_pass"`:**
     - Evaluates $n_{\text{mc}}$ paired draws. Only recommended for linear SCMs; taking $|Y_+ - Y_-|$ per noise draw introduces upward Jensen bias when noise does not cancel.
- **Parameters:**
  - `n_mc` (`int`): Monte Carlo sample size for single-pass mode (default `200_000`).
  - `delta_sd` (`float`): Perturbation factor in units of marginal standard deviation $\sigma_j$ (default `1.0`).
  - `seed` (`int`): Base seed for evaluation.
  - `signed` (`bool`): If `True`, returns signed effect; if `False`, returns mean absolute effect.
  - `n_ref` (`int`): Number of outer reference points drawn from $P(X_j)$ (default `50`).
  - `n_noise_per_ref` (`int`): Inner noise draws per reference point (default `5_000`).
  - `mode` (`str`): `"auto"`, `"nested"`, or `"single_pass"`.
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
- **Behavior:**
  - Uses the exact same $1/\sqrt{k}$ scaling, normalized activations, and clipping as the generative process.
  - Guaranteed invariant: $f^*(X) = Y - \text{noise}_Y$ for all unclipped sampled rows.
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
Allows lossless round-trip serialization of graph topology, structural equation parameters, noise models, standardization settings, and metadata.

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
    chunk_size: int = 8,
    zero_tol: float = 1e-12,
) -> np.ndarray:
```
Computes exact interventional (marginal) Shapley values **fully vectorized over all $2^d$ coalitions simultaneously** via tensor broadcasting and coalition projection matrix multiplication.

- **Vectorized Architecture:**
  - Masks tensor: $(1, 2^d, 1, d)$
  - Background tensor: $(1, 1, n_{\text{bg}}, d)$
  - Chunk tensor: $(n_c, 1, 1, d)$
  - Hybrid tensor: $(n_c, 2^d, n_{\text{bg}}, d)$ evaluated through $f$ in vectorized batches without any Python loop over coalitions.
  - Shapley value projection: $\Phi = V \cdot W$, where $V \in \mathbb{R}^{n_c \times 2^d}$ is the coalition value table and $W \in \mathbb{R}^{2^d \times d}$ is the exact precomputed combinatorial weight matrix.
- **Computational Complexity Note:**
  - For $N_{\text{explain}}$ instances, $N_{\text{bg}}$ background instances, and dimension $d$, the oracle evaluates exactly:
    $$\text{Total evaluations} = N_{\text{explain}} \times N_{\text{bg}} \times 2^d$$
    For $N_{\text{explain}}=200, N_{\text{bg}}=100, d=10$, this represents **20.48 million** model evaluations ($200 \times 100 \times 1024$).
  - **Important for downstream model teams:** Because oracle cost scales directly with the model's prediction throughput, MLP and XGBoost implementations should support fast batched inference.
- **Guaranteed Axioms:**
  - **Efficiency:** $\sum_{j=1}^d \phi_j(x) = f(x) - \mathbb{E}_{X_B}[f(X_B)]$ within $10^{-8}$.
  - **Linear consistency:** For $f(x) = \beta^T x + b$, $\phi_j(x) = \beta_j (x_j - \bar{x}_{B, j})$.
  - **Dummy player:** If feature $j$ is not in the active parents of $f$, $\phi_j(x) = 0$ for all $x$. Values below `zero_tol=1e-12` are set to exact zero to prevent machine-precision residuals from corrupting rank-based metrics.
- **Returns:**
  - `phi`: `(n_explain, d)` array of exact Shapley values.

### 2.2 `mean_abs_shap`

```python
def mean_abs_shap(
    f: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_background: np.ndarray,
    n_explain: int = 100,
    n_background: int = 50,
    max_d: int = 10,
    chunk_size: int = 8,
    zero_tol: float = 1e-12,
) -> np.ndarray:
```
Convenience function returning the mean absolute attribution across explained instances:
$$\bar{\phi}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x^{(i)})|$$

- **Default Parameters:**
  - `n_explain`: `100` instances.
  - `n_background`: `50` instances (total evaluations: $100 \times 50 \times 2^d = 5.12\text{M}$ for $d=10$, completed in $< 1\text{s}$).
- **Returns:**
  - `(d,)` array of mean absolute feature attributions.

---

## 3. Evaluation Metrics: Handling Zeros and Ties

Downstream modules evaluating explanation agreement against causal ground truth ($\tau_j$ vs $\bar{\phi}_j$) must account for features with exact zero attribution:
1. **Kendall $\tau_b$ (`scipy.stats.kendalltau(..., variant='b')`):**
   - Use Kendall $\tau_b$ rather than Spearman rank correlation. Spearman correlation degrades when a vector contains multiple exact zeros (tied ranks), because infinitesimal floating-point noise breaks ties arbitrarily. Kendall $\tau_b$ is explicitly tie-aware and penalizes concordant/discordant pairs appropriately.
2. **Top-$k$ Overlap ($k=3$):**
   - Quantifies the agreement between the top-$k$ causal drivers and the top-$k$ attributed features:
     $$\text{Top-}k \text{ Overlap} = \frac{|\text{Top-}k(\tau) \cap \text{Top-}k(|\Phi|)|}{k}$$
   - Highly interpretable for practitioners who prioritize whether the most critical features are correctly identified.

---

## 4. Dataset Generator CLI (`src.scm.generate`)

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
| `standardize` | `bool` | Whether mechanism scaling and centering were applied (default: `True`) |
| `clip_bound` | `float` | Output clipping boundary (default: `10.0`) |
| `seed` | `int` | Random seed used for generation |
| `split` | `str` | `"train"` or `"heldout"` (grouped by graph family) |
