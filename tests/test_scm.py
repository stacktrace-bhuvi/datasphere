"""Tests for SCM class, intervention engine, and true effect calculations."""

import tempfile
from pathlib import Path
import networkx as nx
import numpy as np
import pytest

from src.scm.scm import SCM
from src.scm.mechanisms import Mechanism, NoiseModel


def test_scm_build_invariants():
    scm = SCM.build("chain_mediation", d=6, seed=42)
    assert nx.is_directed_acyclic_graph(scm.graph)
    assert "Y" in scm.graph.nodes
    assert scm.graph.out_degree("Y") == 0
    assert len(list(scm.graph.predecessors("Y"))) > 0


def test_seed_reproducibility():
    scm = SCM.build("erdos_renyi_dag", d=5, seed=10)
    df1 = scm.sample(1000, seed=99)
    df2 = scm.sample(1000, seed=99)
    df3 = scm.sample(1000, seed=100)

    assert df1.equals(df2)
    assert not df1.equals(df3)


def test_sample_do_root_node_shifts_only_descendants():
    # Build a simple DAG: X1 -> X2 -> Y, X3 -> Y, X4, X5 isolated roots
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_node(f"X{i}")
    G.add_node("Y")
    G.add_edge("X1", "X2")
    G.add_edge("X2", "Y")
    G.add_edge("X3", "Y")

    mechs = {}
    for f in ["X1", "X3", "X4", "X5"]:
        mechs[f] = Mechanism(f, [], "linear", bias=0.0, noise_model=NoiseModel("gaussian", 1.0))
    mechs["X2"] = Mechanism("X2", ["X1"], "linear", weights=[2.0], bias=0.0, noise_model=NoiseModel("gaussian", 0.5))
    mechs["Y"] = Mechanism("Y", ["X2", "X3"], "linear", weights=[1.5, -1.0], bias=0.0, noise_model=NoiseModel("gaussian", 0.5))

    scm = SCM(graph=G, mechanisms=mechs)

    seed = 42
    n = 1000
    df_obs = scm.sample(n, seed=seed)

    # Intervene on root X1
    df_do = scm.sample_do(n, seed=seed, interventions={"X1": 5.0})

    # Non-descendants of X1 are X3, X4, X5
    # They should be EXACTLY identical because of common random numbers
    assert np.allclose(df_obs["X3"].values, df_do["X3"].values)
    assert np.allclose(df_obs["X4"].values, df_do["X4"].values)
    assert np.allclose(df_obs["X5"].values, df_do["X5"].values)

    # X1 was intervened
    assert np.allclose(df_do["X1"].values, 5.0)

    # Descendants of X1 are X2 and Y; they should be shifted
    assert not np.allclose(df_obs["X2"].values, df_do["X2"].values)
    assert not np.allclose(df_obs["Y"].values, df_do["Y"].values)


def test_f_star_evaluation():
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_node(f"X{i}")
    G.add_node("Y")
    G.add_edge("X2", "Y")
    G.add_edge("X4", "Y")

    mechs = {f: Mechanism(f, [], "linear") for f in ["X1", "X2", "X3", "X4", "X5"]}
    mechs["Y"] = Mechanism("Y", ["X2", "X4"], "linear", weights=[3.0, -2.0], bias=1.0)

    scm = SCM(graph=G, mechanisms=mechs)

    X_test = np.array([
        [10.0, 2.0, 99.0, 4.0, 88.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
    ])
    # Y = 3*X2 - 2*X4 + 1
    # row 0: 3(2) - 2(4) + 1 = 6 - 8 + 1 = -1
    # row 1: 3(1) - 2(1) + 1 = 3 - 2 + 1 = 2
    y_eval = scm.f_star(X_test)
    assert np.allclose(y_eval, [-1.0, 2.0])

    # 1D input
    y_1d = scm.f_star(X_test[0])
    assert np.isclose(y_1d, -1.0)


def test_linear_true_effects_matches_analytic():
    # Chain with branching: X1 -> X2 -> Y, X1 -> Y, X3 -> Y, X4, X5
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_node(f"X{i}")
    G.add_node("Y")
    G.add_edge("X1", "X2")
    G.add_edge("X2", "Y")
    G.add_edge("X1", "Y")
    G.add_edge("X3", "Y")

    mechs = {}
    mechs["X1"] = Mechanism("X1", [], "linear", noise_model=NoiseModel("gaussian", 1.0))
    mechs["X2"] = Mechanism("X2", ["X1"], "linear", weights=[1.5], noise_model=NoiseModel("gaussian", 0.8))
    mechs["X3"] = Mechanism("X3", [], "linear", noise_model=NoiseModel("gaussian", 1.2))
    mechs["X4"] = Mechanism("X4", [], "linear", noise_model=NoiseModel("gaussian", 1.0))
    mechs["X5"] = Mechanism("X5", [], "linear", noise_model=NoiseModel("gaussian", 1.0))
    mechs["Y"] = Mechanism("Y", ["X1", "X2", "X3"], "linear", weights=[0.8, -1.2, 2.0], noise_model=NoiseModel("gaussian", 0.5))

    scm = SCM(graph=G, mechanisms=mechs)

    # Compute analytic effects
    analytic = scm.analytic_linear_total_effects(delta_sd=1.0, signed=False)
    # Compute MC true effects with 100_000 samples for test speed
    mc_effects = scm.true_effects(n_mc=100_000, delta_sd=1.0, seed=123, signed=False)

    # Isolated nodes X4 and X5 have effect 0
    assert np.isclose(analytic[3], 0.0)
    assert np.isclose(analytic[4], 0.0)
    assert np.isclose(mc_effects[3], 0.0, atol=1e-5)
    assert np.isclose(mc_effects[4], 0.0, atol=1e-5)

    # All active effects match analytic within 2% relative error or 0.05 absolute
    assert np.allclose(mc_effects, analytic, rtol=0.03, atol=0.05)


def test_scm_serialization_round_trip():
    scm = SCM.build("layered", d=5, seed=77, nonlinearity_prob=0.3, noise_type="laplace", noise_scale=1.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_scm.json"
        scm.to_json(json_path)

        loaded_scm = SCM.from_json(json_path)

        assert loaded_scm.d == scm.d
        assert list(loaded_scm.graph.edges()) == list(scm.graph.edges())
        assert loaded_scm.metadata == scm.metadata

        # Test evaluation matches exactly
        df_orig = scm.sample(500, seed=42)
        df_loaded = loaded_scm.sample(500, seed=42)
        assert df_orig.equals(df_loaded)
