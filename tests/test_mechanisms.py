"""Tests for NoiseModel and Mechanism classes."""

import numpy as np
import pytest
from src.scm.mechanisms import Mechanism, NoiseModel


def test_noise_model_gaussian_and_laplace():
    rng = np.random.default_rng(42)
    nm_g = NoiseModel("gaussian", scale=1.5)
    samples_g = nm_g.sample(10_000, rng)
    assert len(samples_g) == 10_000
    assert np.isclose(np.mean(samples_g), 0.0, atol=0.1)
    assert np.isclose(np.std(samples_g), 1.5, atol=0.1)

    nm_l = NoiseModel("laplace", scale=2.0)
    samples_l = nm_l.sample(10_000, rng)
    assert len(samples_l) == 10_000
    assert np.isclose(np.mean(samples_l), 0.0, atol=0.1)
    assert np.isclose(np.std(samples_l), 2.0, atol=0.1)


def test_noise_model_serialization():
    nm = NoiseModel("laplace", scale=0.8)
    d = nm.to_dict()
    nm_restored = NoiseModel.from_dict(d)
    assert nm_restored.noise_type == "laplace"
    assert nm_restored.scale == 0.8


def test_mechanism_linear_evaluation():
    m = Mechanism(
        node_name="Y",
        parents=["X1", "X2"],
        mechanism_type="linear",
        weights=[2.0, -1.5],
        bias=0.5,
    )
    pvals = {
        "X1": np.array([1.0, 2.0, 3.0]),
        "X2": np.array([0.0, 1.0, 2.0]),
    }
    noiseless = m.evaluate_noiseless(pvals)
    # 2*X1 - 1.5*X2 + 0.5
    # i=0: 2(1) - 0 + 0.5 = 2.5
    # i=1: 2(2) - 1.5(1) + 0.5 = 3.0
    # i=2: 2(3) - 1.5(2) + 0.5 = 3.5
    assert np.allclose(noiseless, [2.5, 3.0, 3.5])


def test_mechanism_nonlinear_additive():
    m = Mechanism(
        node_name="X3",
        parents=["X1", "X2"],
        mechanism_type="nonlinear_additive",
        weights=[1.0, 2.0],
        bias=0.0,
        nonlinearities=["tanh", "relu"],
    )
    pvals = {
        "X1": np.array([0.0, 1.0]),
        "X2": np.array([-2.0, 3.0]),
    }
    noiseless = m.evaluate_noiseless(pvals)
    expected = [np.tanh(0.0) + 2.0 * 0.0, np.tanh(1.0) + 2.0 * 3.0]
    assert np.allclose(noiseless, expected)


def test_mechanism_pairwise_interaction():
    m = Mechanism(
        node_name="Y",
        parents=["X1", "X2"],
        mechanism_type="pairwise_interaction",
        weights=[1.0, 1.0],
        bias=0.0,
        nonlinearities=["linear", "linear"],
        interaction_weights={"X1*X2": 0.5},
    )
    pvals = {
        "X1": np.array([2.0, 3.0]),
        "X2": np.array([4.0, 5.0]),
    }
    noiseless = m.evaluate_noiseless(pvals)
    # i=0: 2 + 4 + 0.5*(8) = 10.0
    # i=1: 3 + 5 + 0.5*(15) = 15.5
    assert np.allclose(noiseless, [10.0, 15.5])


def test_mechanism_serialization_round_trip():
    m = Mechanism(
        node_name="Y",
        parents=["X1", "X2"],
        mechanism_type="pairwise_interaction",
        weights=[1.2, -0.7],
        bias=0.3,
        nonlinearities=["sin", "quadratic"],
        interaction_weights={"X1*X2": 0.4},
        noise_model=NoiseModel("laplace", 0.5),
    )
    d = m.to_dict()
    restored = Mechanism.from_dict(d)
    assert restored.node_name == m.node_name
    assert restored.parents == m.parents
    assert restored.mechanism_type == m.mechanism_type
    assert restored.weights == m.weights
    assert restored.bias == m.bias
    assert restored.nonlinearities == m.nonlinearities
    assert restored.interaction_weights == m.interaction_weights
    assert restored.noise_model.noise_type == "laplace"
    assert restored.noise_model.scale == 0.5
