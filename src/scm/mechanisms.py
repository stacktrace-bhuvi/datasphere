"""Structural equation mechanisms and noise models for SCM nodes."""

from __future__ import annotations
from typing import Any, Callable
import numpy as np


class NoiseModel:
    """Configurable noise generator for structural equations."""

    VALID_TYPES = {"gaussian", "laplace"}

    def __init__(self, noise_type: str = "gaussian", scale: float = 1.0) -> None:
        if noise_type not in self.VALID_TYPES:
            raise ValueError(f"noise_type must be one of {self.VALID_TYPES}, got {noise_type}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.noise_type = noise_type
        self.scale = float(scale)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n noise values using provided random generator."""
        if self.noise_type == "gaussian":
            return rng.normal(loc=0.0, scale=self.scale, size=n)
        elif self.noise_type == "laplace":
            return rng.laplace(loc=0.0, scale=self.scale / np.sqrt(2.0), size=n)
        raise RuntimeError(f"Unsupported noise type: {self.noise_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_type": self.noise_type,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseModel:
        return cls(noise_type=data["noise_type"], scale=data["scale"])


# Supported non-linear transformations
NONLINEAR_MAP: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda x: x,
    "none": lambda x: x,
    "tanh": lambda x: np.tanh(x),
    "sin": lambda x: np.sin(x),
    "quadratic": lambda x: np.square(x),
    "relu": lambda x: np.maximum(0.0, x),
}


class Mechanism:
    """Structural equation mechanism for a single node in an SCM.

    Evaluates:
      X_node = f(parents) + Noise
    where f can be linear, additive nonlinear, or include pairwise interaction.
    """

    VALID_TYPES = {"linear", "nonlinear_additive", "pairwise_interaction"}

    def __init__(
        self,
        node_name: str,
        parents: list[str],
        mechanism_type: str = "linear",
        weights: list[float] | None = None,
        bias: float = 0.0,
        nonlinearities: list[str] | None = None,
        interaction_weights: dict[str, float] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        if mechanism_type not in self.VALID_TYPES:
            raise ValueError(f"mechanism_type must be in {self.VALID_TYPES}, got {mechanism_type}")

        self.node_name = node_name
        self.parents = list(parents)
        self.mechanism_type = mechanism_type
        self.bias = float(bias)

        k = len(self.parents)
        if weights is None:
            self.weights = [1.0] * k
        else:
            if len(weights) != k:
                raise ValueError(f"weights length ({len(weights)}) does not match parents length ({k})")
            self.weights = [float(w) for w in weights]

        if nonlinearities is None:
            self.nonlinearities = ["linear"] * k
        else:
            if len(nonlinearities) != k:
                raise ValueError(f"nonlinearities length ({len(nonlinearities)}) does not match parents ({k})")
            for nl in nonlinearities:
                if nl not in NONLINEAR_MAP:
                    raise ValueError(f"Unsupported nonlinearity '{nl}'. Must be in {list(NONLINEAR_MAP.keys())}")
            self.nonlinearities = list(nonlinearities)

        self.interaction_weights = {}
        if interaction_weights:
            self.interaction_weights = {k_: float(v) for k_, v in interaction_weights.items()}

        self.noise_model = noise_model if noise_model is not None else NoiseModel("gaussian", 1.0)

    def evaluate_noiseless(self, parent_values: dict[str, np.ndarray], n_samples: int | None = None) -> np.ndarray:
        """Vectorized computation of the deterministic component f(parents)."""
        if len(self.parents) == 0:
            if n_samples is None:
                return np.full((1,), self.bias, dtype=np.float64)
            return np.full((n_samples,), self.bias, dtype=np.float64)

        # Get first parent length to determine sample size
        first_parent = self.parents[0]
        n = len(parent_values[first_parent])
        out = np.full(n, self.bias, dtype=np.float64)

        if self.mechanism_type in ("linear", "nonlinear_additive"):
            for parent, w, nl in zip(self.parents, self.weights, self.nonlinearities):
                pval = parent_values[parent]
                fn = NONLINEAR_MAP[nl]
                out += w * fn(pval)
        elif self.mechanism_type == "pairwise_interaction":
            # Add additive component
            for parent, w, nl in zip(self.parents, self.weights, self.nonlinearities):
                pval = parent_values[parent]
                fn = NONLINEAR_MAP[nl]
                out += w * fn(pval)
            # Add pairwise interactions: key format "P1*P2"
            for pair_key, w_pair in self.interaction_weights.items():
                p1, p2 = pair_key.split("*")
                if p1 in parent_values and p2 in parent_values:
                    out += w_pair * (parent_values[p1] * parent_values[p2])

        return out

    def evaluate(
        self,
        parent_values: dict[str, np.ndarray],
        noise: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Vectorized computation of f(parents) + Noise."""
        noiseless = self.evaluate_noiseless(parent_values, n_samples=n_samples)
        n = len(noiseless)

        if noise is not None:
            if len(noise) != n:
                raise ValueError(f"Provided noise length ({len(noise)}) does not match sample count ({n})")
            return noiseless + noise

        if rng is not None:
            noise_draw = self.noise_model.sample(n, rng)
            return noiseless + noise_draw

        # If neither noise nor rng provided, return noiseless
        return noiseless

    def to_dict(self) -> dict[str, Any]:
        """Convert mechanism to a JSON-serializable dictionary."""
        return {
            "node_name": self.node_name,
            "parents": self.parents,
            "mechanism_type": self.mechanism_type,
            "weights": self.weights,
            "bias": self.bias,
            "nonlinearities": self.nonlinearities,
            "interaction_weights": self.interaction_weights,
            "noise_model": self.noise_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mechanism:
        """Instantiate Mechanism from a dictionary."""
        noise_model = NoiseModel.from_dict(data["noise_model"])
        return cls(
            node_name=data["node_name"],
            parents=data["parents"],
            mechanism_type=data["mechanism_type"],
            weights=data["weights"],
            bias=data["bias"],
            nonlinearities=data["nonlinearities"],
            interaction_weights=data.get("interaction_weights"),
            noise_model=noise_model,
        )
