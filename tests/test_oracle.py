"""Tests for exact Shapley value oracle and estimand gap demonstration."""

import networkx as nx
import numpy as np
import pytest

from src.scm.oracle import exact_shap, mean_abs_shap
from src.scm.scm import SCM
from src.scm.mechanisms import Mechanism, NoiseModel


def test_exact_shap_efficiency():
    """Verify efficiency axiom: sum_j phi_j(x) = f(x) - E_{X_B}[f(X_B)] within 1e-8."""
    rng = np.random.default_rng(42)
    d = 5
    X_exp = rng.normal(0, 1, size=(20, d))
    X_bg = rng.normal(0, 1, size=(50, d))

    # Nonlinear function with interactions
    def f(X: np.ndarray) -> np.ndarray:
        return (
            1.5 * X[:, 0]
            - 2.0 * np.tanh(X[:, 1])
            + 0.8 * np.square(X[:, 2])
            + 1.2 * X[:, 3] * X[:, 4]
            + 0.5
        )

    phi = exact_shap(f, X_exp, X_bg, max_d=10)
    assert phi.shape == (20, d)

    sum_phi = np.sum(phi, axis=1)
    f_exp = f(X_exp)
    mean_f_bg = np.mean(f(X_bg))
    expected_sum = f_exp - mean_f_bg

    assert np.allclose(sum_phi, expected_sum, atol=1e-8)


def test_exact_shap_linear_model():
    """For a linear model f(x) = beta^T x + b, phi_j(x) = beta_j * (x_j - mean(x_bg_j))."""
    rng = np.random.default_rng(101)
    d = 6
    betas = np.array([2.5, -1.8, 0.0, 3.2, -0.5, 1.1])
    bias = 4.0

    def f_linear(X: np.ndarray) -> np.ndarray:
        return X @ betas + bias

    X_exp = rng.normal(1.0, 2.0, size=(15, d))
    X_bg = rng.normal(0.0, 1.5, size=(40, d))

    phi = exact_shap(f_linear, X_exp, X_bg, max_d=10)

    # Analytic Shapley values
    bg_means = np.mean(X_bg, axis=0)  # (d,)
    expected_phi = (X_exp - bg_means) * betas  # (15, d)

    assert np.allclose(phi, expected_phi, atol=1e-8)


def test_exact_shap_chunking_consistency():
    rng = np.random.default_rng(7)
    d = 5
    X_exp = rng.normal(0, 1, size=(30, d))
    X_bg = rng.normal(0, 1, size=(20, d))

    def f(X: np.ndarray) -> np.ndarray:
        return np.sin(X[:, 0]) + X[:, 1] * X[:, 2]

    phi_chunk_10 = exact_shap(f, X_exp, X_bg, max_d=10, chunk_size=10)
    phi_chunk_64 = exact_shap(f, X_exp, X_bg, max_d=10, chunk_size=64)

    assert np.allclose(phi_chunk_10, phi_chunk_64, atol=1e-12)


def test_mediation_estimand_gap():
    """Mediation check demonstrating the fundamental estimand gap:

    In a chain X1 -> X2 -> Y (with d=5 features, X1 not directly connected to Y):
      - f_star is Y = g(X2), so f_star has no dependence on X1.
      - Exact SHAP on f_star gives phi_1 = 0 for all instances (dummy player).
      - However, intervening on X1 via do(X1 = x) causes X2 to shift, which in turn
        shifts Y. Hence the true causal interventional effect tau_1 > 0!
      - This illustrates the structural incompatibility between feature attribution
        on f* and the true interventional estimand.
    """
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_node(f"X{i}")
    G.add_node("Y")

    # Chain: X1 -> X2 -> Y, with X3, X4, X5 isolated
    G.add_edge("X1", "X2")
    G.add_edge("X2", "Y")

    mechs = {}
    mechs["X1"] = Mechanism("X1", [], "linear", noise_model=NoiseModel("gaussian", 1.0))
    mechs["X2"] = Mechanism("X2", ["X1"], "linear", weights=[2.0], noise_model=NoiseModel("gaussian", 0.5))
    for f in ["X3", "X4", "X5"]:
        mechs[f] = Mechanism(f, [], "linear", noise_model=NoiseModel("gaussian", 1.0))
    # Y depends directly ONLY on X2
    mechs["Y"] = Mechanism("Y", ["X2"], "linear", weights=[1.5], bias=0.0, noise_model=NoiseModel("gaussian", 0.5))

    scm = SCM(graph=G, mechanisms=mechs)

    # 1. Compute true causal interventional effects tau
    tau = scm.true_effects(n_mc=100_000, delta_sd=1.0, seed=42, signed=False)

    # In this chain, X1 has a strong indirect effect on Y:
    # Path X1 -> X2 -> Y: weight 2.0 * 1.5 = 3.0
    assert tau[0] > 1.0  # X1 has significant true interventional effect!
    assert tau[1] > 1.0  # X2 has direct effect on Y
    assert np.isclose(tau[2], 0.0, atol=1e-4)  # X3 is isolated
    assert np.isclose(tau[3], 0.0, atol=1e-4)  # X4 is isolated
    assert np.isclose(tau[4], 0.0, atol=1e-4)  # X5 is isolated

    # 2. Compute exact SHAP on the true generating function f_star
    obs = scm.sample(1000, seed=42)
    feature_cols = [f"X{i}" for i in range(1, 6)]
    X_exp = obs[feature_cols].values[:100]
    X_bg = obs[feature_cols].values[100:300]

    phi_fstar = exact_shap(scm.f_star, X_exp, X_bg, max_d=10)
    mean_abs_phi = np.mean(np.abs(phi_fstar), axis=0)

    # f_star only reads X2. For X1 (column 0), f_star is completely invariant:
    # f_star(x1, x2, x3, x4, x5) = 1.5 * x2
    # Therefore, X1 is a null/dummy player for f_star, so phi_1 MUST BE 0.0!
    assert np.allclose(phi_fstar[:, 0], 0.0, atol=1e-12)
    assert np.isclose(mean_abs_phi[0], 0.0, atol=1e-12)

    # BUT tau[0] is strictly non-zero!
    # Estimand gap for X1: |tau_1 - mean|phi_1(f*)|| = tau_1 > 0
    estimand_gap_x1 = abs(tau[0] - mean_abs_phi[0])
    assert estimand_gap_x1 > 1.0
