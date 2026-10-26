"""Structural Causal Model (SCM) core class with causal intervention engine."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd

from src.scm.mechanisms import Mechanism, NoiseModel
from src.scm.graph import (
    generate_dag,
    validate_dag,
    compute_graph_metrics,
    GRAPH_FAMILIES,
)


class SCM:
    """Structural Causal Model over d features X1..Xd and sink outcome Y.

    Guarantees:
      - Graph is a validated DAG.
      - Y is a strict sink node (out-degree 0).
      - All structural equations are vectorized callables.
      - Exact common random numbers (CRN) between sample() and sample_do().
      - Exact serialization to and from JSON.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        mechanisms: dict[str, Mechanism],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.d = graph.graph.get("d", len([n for n in graph.nodes if n != "Y"]))
        validate_dag(graph, self.d)

        self.graph = graph
        self.features = [f"X{i}" for i in range(1, self.d + 1)]
        self.nodes = self.features + ["Y"]

        for node in self.nodes:
            if node not in mechanisms:
                raise ValueError(f"Missing mechanism for node '{node}'")
        self.mechanisms = mechanisms

        # Cache topological ordering
        self.topological_order = list(nx.topological_sort(self.graph))
        self.metadata = metadata if metadata is not None else {}

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
        """Procedurally build an SCM with guaranteed invariants."""
        G = generate_dag(family, d, seed)
        rng = np.random.default_rng(seed)

        mechanisms: dict[str, Mechanism] = {}
        nonlinear_options = ["tanh", "sin", "quadratic", "relu"]

        for node in G.nodes:
            parents = sorted(list(G.predecessors(node)))
            k = len(parents)

            # Node noise
            n_scale = float(noise_scale * rng.uniform(0.8, 1.2))
            node_noise = NoiseModel(noise_type=noise_type, scale=n_scale)

            if k == 0:
                # Root node: standard linear with noise
                mechanisms[node] = Mechanism(
                    node_name=node,
                    parents=[],
                    mechanism_type="linear",
                    bias=float(rng.normal(0.0, 0.5)),
                    noise_model=node_noise,
                )
            else:
                # Weights drawn uniformly in [-1.5, -0.5] U [0.5, 1.5]
                signs = rng.choice([-1.0, 1.0], size=k)
                mags = rng.uniform(0.5, 1.5, size=k)
                weights = [float(s * m) for s, m in zip(signs, mags)]
                bias = float(rng.normal(0.0, 0.2))

                is_nonlinear = rng.random() < nonlinearity_prob
                has_interaction = (k >= 2) and (rng.random() < interaction_prob)

                if has_interaction:
                    # Choose a pair for interaction
                    p1, p2 = parents[0], parents[1]
                    interaction_weight = float(rng.uniform(0.3, 0.8) * rng.choice([-1.0, 1.0]))
                    nl_types = ["linear"] * k
                    mechanisms[node] = Mechanism(
                        node_name=node,
                        parents=parents,
                        mechanism_type="pairwise_interaction",
                        weights=weights,
                        bias=bias,
                        nonlinearities=nl_types,
                        interaction_weights={f"{p1}*{p2}": interaction_weight},
                        noise_model=node_noise,
                    )
                elif is_nonlinear:
                    nl_types = [str(rng.choice(nonlinear_options)) for _ in range(k)]
                    mechanisms[node] = Mechanism(
                        node_name=node,
                        parents=parents,
                        mechanism_type="nonlinear_additive",
                        weights=weights,
                        bias=bias,
                        nonlinearities=nl_types,
                        noise_model=node_noise,
                    )
                else:
                    mechanisms[node] = Mechanism(
                        node_name=node,
                        parents=parents,
                        mechanism_type="linear",
                        weights=weights,
                        bias=bias,
                        nonlinearities=["linear"] * k,
                        noise_model=node_noise,
                    )

        metrics = compute_graph_metrics(G, d)
        n_nonlinear = sum(
            1 for m in mechanisms.values() if m.mechanism_type != "linear"
        )

        metadata = {
            "family": family,
            "d": d,
            "seed": seed,
            "n_nonlinear": n_nonlinear,
            "noise_type": noise_type,
            "noise_scale": noise_scale,
            "has_confounding": metrics["has_confounding"],
            "mediation_share": metrics["mediation_share"],
        }

        return cls(graph=G, mechanisms=mechanisms, metadata=metadata)

    def _draw_noise(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Draw deterministic noise for all nodes using common random numbers."""
        noise_draws: dict[str, np.ndarray] = {}
        for idx, node in enumerate(self.nodes):
            # Deterministic per-node RNG derived from master seed
            node_rng = np.random.default_rng([seed, idx + 1000])
            noise_draws[node] = self.mechanisms[node].noise_model.sample(n, node_rng)
        return noise_draws

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample n observational draws from the SCM."""
        noise_draws = self._draw_noise(n, seed)
        values: dict[str, np.ndarray] = {}

        for node in self.topological_order:
            mech = self.mechanisms[node]
            parent_vals = {p: values[p] for p in mech.parents}
            values[node] = mech.evaluate(parent_vals, noise=noise_draws[node], n_samples=n)

        cols = self.features + ["Y"]
        return pd.DataFrame({c: values[c] for c in cols})

    def sample_do(
        self,
        n: int,
        seed: int,
        interventions: dict[str, float | np.ndarray],
    ) -> pd.DataFrame:
        """Sample under interventional do() operations using common random numbers (CRN).

        Non-descendants of intervened nodes remain identical to observational sample(n, seed).
        Descendants are recomputed using the exact same noise draws.
        """
        for k in interventions:
            if k not in self.nodes:
                raise ValueError(f"Intervened node '{k}' is not part of the SCM.")

        noise_draws = self._draw_noise(n, seed)
        values: dict[str, np.ndarray] = {}

        # Pre-broadcast intervention arrays
        interv_arrays: dict[str, np.ndarray] = {}
        for k, v in interventions.items():
            if np.isscalar(v):
                interv_arrays[k] = np.full(n, float(v), dtype=np.float64)
            else:
                arr = np.asarray(v, dtype=np.float64)
                if len(arr) != n:
                    raise ValueError(f"Intervention array for '{k}' must have length {n}, got {len(arr)}")
                interv_arrays[k] = arr

        for node in self.topological_order:
            if node in interv_arrays:
                values[node] = interv_arrays[node]
            else:
                mech = self.mechanisms[node]
                parent_vals = {p: values[p] for p in mech.parents}
                values[node] = mech.evaluate(parent_vals, noise=noise_draws[node], n_samples=n)

        cols = self.features + ["Y"]
        return pd.DataFrame({c: values[c] for c in cols})

    def f_star(self, X: np.ndarray) -> np.ndarray:
        """Noiseless true structural equation for outcome Y.

        Takes full feature matrix of shape (n, d) or (d,), ignoring non-parents of Y.
        Feature j (0-indexed) corresponds to X_{j+1}.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = False
        if X_arr.ndim == 1:
            is_1d = True
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} feature columns, got {X_arr.shape[1]}")

        # Map feature names to column arrays
        feat_map = {f"X{i+1}": X_arr[:, i] for i in range(self.d)}
        y_mech = self.mechanisms["Y"]
        parent_vals = {p: feat_map[p] for p in y_mech.parents}

        out = y_mech.evaluate_noiseless(parent_vals, n_samples=len(X_arr))
        return out[0] if is_1d else out

    def true_effects(
        self,
        n_mc: int = 200_000,
        delta_sd: float = 1.0,
        seed: int = 0,
        signed: bool = False,
    ) -> np.ndarray:
        """Compute exact total interventional effect tau_j for each feature X_j.

        tau_j = mean over reference values x drawn from marginal of X_j of:
          |E[Y | do(X_j = x + delta_sd * sd_j)] - E[Y | do(X_j = x - delta_sd * sd_j)]| / 2
        (or signed variant if signed=True).
        Uses common random numbers (CRN).
        """
        # Step 1: Draw observational reference points to sample from P(X_j) and compute sd_j
        obs = self.sample(n=n_mc, seed=seed)
        effects = np.zeros(self.d, dtype=np.float64)

        for j_idx, feat in enumerate(self.features):
            sd_j = float(obs[feat].std())
            if sd_j < 1e-12:
                effects[j_idx] = 0.0
                continue

            x_ref = obs[feat].values
            shift = delta_sd * sd_j

            # Using CRN with seed + 1 for the intervention runs
            crn_seed = seed + 42
            df_plus = self.sample_do(n_mc, seed=crn_seed, interventions={feat: x_ref + shift})
            df_minus = self.sample_do(n_mc, seed=crn_seed, interventions={feat: x_ref - shift})

            diff = (df_plus["Y"].values - df_minus["Y"].values) / 2.0
            if signed:
                effects[j_idx] = float(np.mean(diff))
            else:
                effects[j_idx] = float(np.mean(np.abs(diff)))

        return effects

    def analytic_linear_total_effects(self, delta_sd: float = 1.0, signed: bool = False) -> np.ndarray:
        """Compute exact analytic total effects for purely linear SCMs.

        In a linear SCM, total effect = sum over directed paths of weight products * (delta_sd * sd_j).
        """
        # Check all mechanisms are linear
        for m in self.mechanisms.values():
            if m.mechanism_type != "linear":
                raise ValueError("analytic_linear_total_effects only supported for linear SCMs.")

        # Estimate marginal standard deviations from a clean observational run
        obs = self.sample(n=50_000, seed=0)
        sds = [float(obs[f].std()) for f in self.features]

        effects = np.zeros(self.d, dtype=np.float64)

        for j_idx, feat in enumerate(self.features):
            sd_j = sds[j_idx]
            if not nx.has_path(self.graph, feat, "Y"):
                effects[j_idx] = 0.0
                continue

            # Sum over all directed paths of product of edge weights
            total_path_weight = 0.0
            for path in nx.all_simple_paths(self.graph, source=feat, target="Y"):
                path_weight = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    mech_v = self.mechanisms[v]
                    p_idx = mech_v.parents.index(u)
                    path_weight *= mech_v.weights[p_idx]
                total_path_weight += path_weight

            tau = total_path_weight * delta_sd * sd_j
            effects[j_idx] = tau if signed else abs(tau)

        return effects

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete SCM specification to a dictionary."""
        return {
            "d": self.d,
            "edges": list(self.graph.edges()),
            "graph_attributes": dict(self.graph.graph),
            "metadata": self.metadata,
            "mechanisms": {k: m.to_dict() for k, m in self.mechanisms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCM:
        """Re-instantiate SCM from a serialized dictionary."""
        d = data["d"]
        G = nx.DiGraph()
        features = [f"X{i}" for i in range(1, d + 1)]
        G.add_nodes_from(features)
        G.add_node("Y")
        G.add_edges_from(data["edges"])
        G.graph.update(data.get("graph_attributes", {}))

        mechanisms = {k: Mechanism.from_dict(v) for k, v in data["mechanisms"].items()}
        return cls(graph=G, mechanisms=mechanisms, metadata=data.get("metadata"))

    def to_json(self, filepath: str | Path) -> None:
        """Save SCM to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str | Path) -> SCM:
        """Load SCM from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
