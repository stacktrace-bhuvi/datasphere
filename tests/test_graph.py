"""Tests for DAG generators and graph structure analysis."""

import pytest
import networkx as nx
from src.scm.graph import (
    GRAPH_FAMILIES,
    generate_dag,
    compute_graph_metrics,
    validate_dag,
)


@pytest.mark.parametrize("family", GRAPH_FAMILIES)
@pytest.mark.parametrize("d", [5, 7, 10])
def test_generate_dag_invariants(family: str, d: int):
    seed = 123
    G = generate_dag(family, d, seed)

    # 1. Must be a DAG
    assert nx.is_directed_acyclic_graph(G)

    # 2. Y is in graph and is a strict sink
    assert "Y" in G.nodes
    assert G.out_degree("Y") == 0

    # 3. All Y parents are among X1..Xd
    expected_features = {f"X{i}" for i in range(1, d + 1)}
    y_parents = set(G.predecessors("Y"))
    assert len(y_parents) > 0
    assert y_parents.issubset(expected_features)

    # 4. Check metadata
    assert G.graph["family"] == family
    assert G.graph["d"] == d
    assert G.graph["seed"] == seed


def test_seed_determinism():
    G1 = generate_dag("chain_mediation", d=6, seed=42)
    G2 = generate_dag("chain_mediation", d=6, seed=42)
    assert list(G1.edges()) == list(G2.edges())

    G3 = generate_dag("chain_mediation", d=6, seed=99)
    # Different seeds should typically differ
    assert list(G1.edges()) != list(G3.edges())


def test_invalid_d():
    with pytest.raises(ValueError, match="d must be between 5 and 10"):
        generate_dag("erdos_renyi_dag", d=4, seed=0)

    with pytest.raises(ValueError, match="d must be between 5 and 10"):
        generate_dag("erdos_renyi_dag", d=11, seed=0)


def test_compute_graph_metrics():
    # Construct a known chain: X1 -> X2 -> Y (with d=5, X3, X4, X5)
    G = nx.DiGraph()
    for i in range(1, 6):
        G.add_node(f"X{i}")
    G.add_node("Y")
    G.add_edge("X1", "X2")
    G.add_edge("X2", "Y")

    metrics = compute_graph_metrics(G, d=5)
    # X1 has path X1 -> X2 -> Y (len 3 > 2), so X1 is mediated
    assert metrics["mediation_share"] == 0.2  # 1 out of 5
    assert metrics["n_y_parents"] == 1
    assert metrics["total_edges"] == 2
