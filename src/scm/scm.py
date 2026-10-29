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

    @property
    def is_linear_scm(self) -> bool:
        """Check if all node mechanisms in the SCM are linear."""
        return all(m.mechanism_type == "linear" for m in self.mechanisms.values())

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
        """Procedurally build an SCM with guaranteed invariants and variance preservation."""
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
                # Root node
                n_scale = float(noise_scale * rng.uniform(0.8, 1.2))
                mechanisms[node] = Mechanism(
                    node_name=node,
                    parents=[],
                    mechanism_type="linear",
                    bias=float(rng.normal(0.0, 0.3)),
                    noise_model=NoiseModel(noise_type=noise_type, scale=n_scale),
                    standardize=standardize,
                    clip_bound=clip_bound,
                )
            else:
                signs = rng.choice([-1.0, 1.0], size=k)
                mags = rng.uniform(0.5, 1.2 if standardize else 1.5, size=k)
                raw_w = signs * mags
                bias = float(rng.normal(0.0, 0.2))

                is_nonlinear = rng.random() < nonlinearity_prob
                has_interaction = (k >= 2) and (rng.random() < interaction_prob)

                if standardize:
                    # Noise scale adjusted to balance signal power: Var(signal) + Var(noise) ~ 1
                    n_scale = float(noise_scale * rng.uniform(0.5, 0.7))
                    node_noise = NoiseModel(noise_type=noise_type, scale=n_scale)

                    if has_interaction:
                        p1, p2 = parents[0], parents[1]
                        raw_inter = float(rng.uniform(0.2, 0.4) * rng.choice([-1.0, 1.0]))
                        norm = float(np.sqrt(np.sum(raw_w**2) + raw_inter**2))
                        weights = [float((w / norm) * 0.75) for w in raw_w]
                        inter_w = float((raw_inter / norm) * 0.35)
                        mechanisms[node] = Mechanism(
                            node_name=node,
                            parents=parents,
                            mechanism_type="pairwise_interaction",
                            weights=weights,
                            bias=bias,
                            nonlinearities=["linear"] * k,
                            interaction_weights={f"{p1}*{p2}": inter_w},
                            noise_model=node_noise,
                            standardize=standardize,
                            clip_bound=clip_bound,
                        )
                    elif is_nonlinear:
                        norm = float(np.sqrt(np.sum(raw_w**2)))
                        weights = [float((w / norm) * 0.75) for w in raw_w]
                        nl_types = [str(rng.choice(nonlinear_options)) for _ in range(k)]
                        mechanisms[node] = Mechanism(
                            node_name=node,
                            parents=parents,
                            mechanism_type="nonlinear_additive",
                            weights=weights,
                            bias=bias,
                            nonlinearities=nl_types,
                            noise_model=node_noise,
                            standardize=standardize,
                            clip_bound=clip_bound,
                        )
                    else:
                        norm = float(np.sqrt(np.sum(raw_w**2)))
                        weights = [float((w / norm) * 0.75) for w in raw_w]
                        mechanisms[node] = Mechanism(
                            node_name=node,
                            parents=parents,
                            mechanism_type="linear",
                            weights=weights,
                            bias=bias,
                            nonlinearities=["linear"] * k,
                            noise_model=node_noise,
                            standardize=standardize,
                            clip_bound=clip_bound,
                        )
                else:
                    n_scale = float(noise_scale * rng.uniform(0.8, 1.2))
                    node_noise = NoiseModel(noise_type=noise_type, scale=n_scale)
                    weights = [float(w) for w in raw_w]

                    if has_interaction:
                        p1, p2 = parents[0], parents[1]
                        interaction_weight = float(rng.uniform(0.3, 0.8) * rng.choice([-1.0, 1.0]))
                        mechanisms[node] = Mechanism(
                            node_name=node,
                            parents=parents,
                            mechanism_type="pairwise_interaction",
                            weights=weights,
                            bias=bias,
                            nonlinearities=["linear"] * k,
                            interaction_weights={f"{p1}*{p2}": interaction_weight},
                            noise_model=node_noise,
                            standardize=standardize,
                            clip_bound=clip_bound,
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
                            standardize=standardize,
                            clip_bound=clip_bound,
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
                            standardize=standardize,
                            clip_bound=clip_bound,
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
            "standardize": standardize,
            "clip_bound": clip_bound,
        }

        return cls(graph=G, mechanisms=mechanisms, metadata=metadata)

    def _draw_noise(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Draw deterministic noise for all nodes using common random numbers."""
        noise_draws: dict[str, np.ndarray] = {}
        for idx, node in enumerate(self.nodes):
            node_rng = np.random.default_rng([seed, idx + 1000])
            noise_draws[node] = self.mechanisms[node].noise_model.sample(n, node_rng)
        return noise_draws

    def sample(self, n: int, seed: int, return_noise: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Sample n observational draws from the SCM."""
        noise_draws = self._draw_noise(n, seed)
        values: dict[str, np.ndarray] = {}

        for node in self.topological_order:
            mech = self.mechanisms[node]
            parent_vals = {p: values[p] for p in mech.parents}
            values[node] = mech.evaluate(parent_vals, noise=noise_draws[node], n_samples=n)

        cols = self.features + ["Y"]
        df = pd.DataFrame({c: values[c] for c in cols})
        if return_noise:
            return df, noise_draws
        return df

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
        Applies identical standardization scaling and clipping as the generative process.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = False
        if X_arr.ndim == 1:
            is_1d = True
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.d:
            raise ValueError(f"Expected input with {self.d} feature columns, got {X_arr.shape[1]}")

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
        n_ref: int = 50,
        n_noise_per_ref: int = 5_000,
        mode: str = "auto",
    ) -> np.ndarray:
        """Compute exact total interventional effect tau_j for each feature X_j.

        tau_j = mean over reference values x drawn from marginal of X_j of:
          |E[Y | do(X_j = x + delta_sd * sd_j)] - E[Y | do(X_j = x - delta_sd * sd_j)]| / 2
        (or signed variant if signed=True).

        Evaluation modes:
          - 'auto' (default): Uses exact single-pass CRN if all mechanisms are linear (where noise
            strictly cancels out). For nonlinear or interaction SCMs, uses the nested estimator
            to prevent Jensen's inequality bias.
          - 'nested': Explicitly averages over n_noise_per_ref noise draws BEFORE taking absolute value.
          - 'single_pass': Evaluates single-pass n_mc paired draws. Note: taking |Y+ - Y-| per noise draw
            is biased upward for nonlinear mechanisms.
        """
        if mode not in ("auto", "nested", "single_pass"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'auto', 'nested', or 'single_pass'.")

        use_nested = (mode == "nested") or (mode == "auto" and not self.is_linear_scm)

        effects = np.zeros(self.d, dtype=np.float64)

        if use_nested:
            obs_ref = self.sample(n=max(n_ref, 5000), seed=seed)
            crn_seed = seed + 42
            total_n = n_ref * n_noise_per_ref

            for j_idx, feat in enumerate(self.features):
                sd_j = float(obs_ref[feat].std())
                if sd_j < 1e-12:
                    effects[j_idx] = 0.0
                    continue

                ref_pts = obs_ref[feat].values[:n_ref]
                shift = delta_sd * sd_j
                ref_expanded_plus = np.repeat(ref_pts + shift, n_noise_per_ref)
                ref_expanded_minus = np.repeat(ref_pts - shift, n_noise_per_ref)

                df_plus = self.sample_do(total_n, seed=crn_seed, interventions={feat: ref_expanded_plus})
                df_minus = self.sample_do(total_n, seed=crn_seed, interventions={feat: ref_expanded_minus})

                # Compute E[Y | do(x + delta)] and E[Y | do(x - delta)] per reference point
                y_plus = df_plus["Y"].values.reshape(n_ref, n_noise_per_ref).mean(axis=1)
                y_minus = df_minus["Y"].values.reshape(n_ref, n_noise_per_ref).mean(axis=1)

                diff = (y_plus - y_minus) / 2.0
                if signed:
                    effects[j_idx] = float(np.mean(diff))
                else:
                    effects[j_idx] = float(np.mean(np.abs(diff)))
        else:
            # Single-pass 200,000 CRN estimation
            obs = self.sample(n=n_mc, seed=seed)
            crn_seed = seed + 42

            for j_idx, feat in enumerate(self.features):
                sd_j = float(obs[feat].std())
                if sd_j < 1e-12:
                    effects[j_idx] = 0.0
                    continue

                x_ref = obs[feat].values
                shift = delta_sd * sd_j

                df_plus = self.sample_do(n_mc, seed=crn_seed, interventions={feat: x_ref + shift})
                df_minus = self.sample_do(n_mc, seed=crn_seed, interventions={feat: x_ref - shift})

                diff = (df_plus["Y"].values - df_minus["Y"].values) / 2.0
                if signed:
                    effects[j_idx] = float(np.mean(diff))
                else:
                    effects[j_idx] = float(np.mean(np.abs(diff)))

        return effects

    def analytic_linear_total_effects(self, delta_sd: float = 1.0, signed: bool = False) -> np.ndarray:
        """Compute exact analytic total effects for purely linear SCMs."""
        for m in self.mechanisms.values():
            if m.mechanism_type != "linear":
                raise ValueError("analytic_linear_total_effects only supported for linear SCMs.")

        obs = self.sample(n=50_000, seed=0)
        sds = [float(obs[f].std()) for f in self.features]

        effects = np.zeros(self.d, dtype=np.float64)

        for j_idx, feat in enumerate(self.features):
            sd_j = sds[j_idx]
            if not nx.has_path(self.graph, feat, "Y"):
                effects[j_idx] = 0.0
                continue

            total_path_weight = 0.0
            for path in nx.all_simple_paths(self.graph, source=feat, target="Y"):
                path_weight = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    mech_v = self.mechanisms[v]
                    p_idx = mech_v.parents.index(u)
                    scale = 1.0 / np.sqrt(len(mech_v.parents)) if mech_v.standardize else 1.0
                    path_weight *= (mech_v.weights[p_idx] * scale)
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
