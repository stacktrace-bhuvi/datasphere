"""Seeded DAG generators and graph structure analysis for SCMs."""

from __future__ import annotations
from typing import Callable, Any
import networkx as nx
import numpy as np


GRAPH_FAMILIES = [
    "chain_mediation",
    "confounded_fork",
    "feature_collider",
    "erdos_renyi_dag",
    "layered",
]


def validate_dag(G: nx.DiGraph, d: int) -> None:
    """Validate that G satisfies project DAG invariants."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains directed cycles; must be a DAG.")

    expected_features = {f"X{i}" for i in range(1, d + 1)}
    nodes = set(G.nodes())
    if "Y" not in nodes:
        raise ValueError("Graph must contain sink node 'Y'.")
    if not expected_features.issubset(nodes):
        raise ValueError(f"Graph nodes must contain all features {expected_features}.")

    if G.out_degree("Y") != 0:
        raise ValueError("Outcome node 'Y' must be a strict sink (no children).")

    y_parents = set(G.predecessors("Y"))
    if not y_parents.issubset(expected_features):
        raise ValueError(f"All parents of Y must be features, got: {y_parents - expected_features}")

    if len(y_parents) == 0:
        raise ValueError("Outcome node 'Y' must have at least one parent feature.")


def generate_chain_mediation_dag(d: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a DAG featuring a strong mediation chain."""
    G = nx.DiGraph()
    features = [f"X{i}" for i in range(1, d + 1)]
    G.add_nodes_from(features)
    G.add_node("Y")

    # Primary mediation chain: X1 -> X2 -> ... -> X_k where k >= 2
    # Ensure X1 is not a direct parent of Y, but X2 (or X_k) is.
    chain_len = min(4, d - 1)
    for i in range(chain_len - 1):
        G.add_edge(features[i], features[i + 1])
    
    # End of chain connects to Y
    G.add_edge(features[chain_len - 1], "Y")

    # Connect remaining features: some to Y, some to the chain
    for i in range(chain_len, d):
        # randomly connect to an earlier node or Y
        target_options = features[:i] + ["Y"]
        n_edges = rng.integers(1, min(3, len(target_options) + 1))
        chosen = rng.choice(target_options, size=n_edges, replace=False)
        for target in chosen:
            if target == "Y":
                G.add_edge(features[i], "Y")
            else:
                G.add_edge(features[i], target)

    # Ensure Y has at least one parent
    if G.in_degree("Y") == 0:
        G.add_edge(features[chain_len - 1], "Y")

    validate_dag(G, d)
    return G


def generate_confounded_fork_dag(d: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a DAG featuring confounding forks (X_confounder -> X_treat and X_confounder -> Y)."""
    G = nx.DiGraph()
    features = [f"X{i}" for i in range(1, d + 1)]
    G.add_nodes_from(features)
    G.add_node("Y")

    # X1 is a major confounder: X1 -> X2 and X1 -> Y
    G.add_edge("X1", "X2")
    G.add_edge("X1", "Y")
    # X2 may or may not point to Y
    if rng.random() > 0.3:
        G.add_edge("X2", "Y")

    # Connect other features to create forks or downstream effects
    for i in range(2, d):
        earlier = features[:i]
        # X_i gets inputs from earlier features
        n_parents = rng.integers(1, min(3, len(earlier) + 1))
        chosen_parents = rng.choice(earlier, size=n_parents, replace=False)
        for p in chosen_parents:
            G.add_edge(p, features[i])

        # Randomly connect to Y
        if rng.random() < 0.5:
            G.add_edge(features[i], "Y")

    if G.in_degree("Y") == 0:
        G.add_edge("X1", "Y")

    validate_dag(G, d)
    return G


def generate_feature_collider_dag(d: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a DAG featuring collider structures among features (X1 -> X3 <- X2)."""
    G = nx.DiGraph()
    features = [f"X{i}" for i in range(1, d + 1)]
    G.add_nodes_from(features)
    G.add_node("Y")

    # Collider: X1 -> X3 and X2 -> X3
    G.add_edge("X1", "X3")
    G.add_edge("X2", "X3")
    # Collider points towards Y or downstream
    G.add_edge("X3", "Y")

    for i in range(3, d):
        earlier = features[:i]
        n_parents = rng.integers(1, min(3, len(earlier) + 1))
        chosen_parents = rng.choice(earlier, size=n_parents, replace=False)
        for p in chosen_parents:
            G.add_edge(p, features[i])

        if rng.random() < 0.4:
            G.add_edge(features[i], "Y")

    if G.in_degree("Y") == 0:
        G.add_edge("X3", "Y")

    validate_dag(G, d)
    return G


def generate_erdos_renyi_dag(d: int, rng: np.random.Generator, edge_prob: float = 0.35) -> nx.DiGraph:
    """Generate an Erdős–Rényi DAG by sampling upper-triangular edges."""
    G = nx.DiGraph()
    features = [f"X{i}" for i in range(1, d + 1)]
    G.add_nodes_from(features)
    G.add_node("Y")

    # Edges between features: i < j
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                G.add_edge(features[i], features[j])

    # Edges from features to Y
    for f in features:
        if rng.random() < edge_prob:
            G.add_edge(f, "Y")

    # Guarantee Y has at least one parent
    if G.in_degree("Y") == 0:
        chosen = rng.choice(features)
        G.add_edge(chosen, "Y")

    validate_dag(G, d)
    return G


def generate_layered_dag(d: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a layered DAG where edges flow strictly from lower to higher layers."""
    G = nx.DiGraph()
    features = [f"X{i}" for i in range(1, d + 1)]
    G.add_nodes_from(features)
    G.add_node("Y")

    # Partition features into 3 layers
    shuffled = list(features)
    rng.shuffle(shuffled)
    n_l1 = max(1, d // 3)
    n_l2 = max(1, d // 3)
    l1 = shuffled[:n_l1]
    l2 = shuffled[n_l1 : n_l1 + n_l2]
    l3 = shuffled[n_l1 + n_l2 :]

    # Edges from L1 to L2
    for u in l1:
        for v in l2:
            if rng.random() < 0.5:
                G.add_edge(u, v)

    # Edges from L2 to L3
    for u in l2:
        for v in l3:
            if rng.random() < 0.5:
                G.add_edge(u, v)

    # Edges from L3 (and some L2) to Y
    for u in l3:
        if rng.random() < 0.7:
            G.add_edge(u, "Y")
    for u in l2:
        if rng.random() < 0.3:
            G.add_edge(u, "Y")

    if G.in_degree("Y") == 0:
        chosen = rng.choice(l3 if l3 else l2)
        G.add_edge(chosen, "Y")

    validate_dag(G, d)
    return G


FAMILY_GENERATORS: dict[str, Callable[[int, np.random.Generator], nx.DiGraph]] = {
    "chain_mediation": generate_chain_mediation_dag,
    "confounded_fork": generate_confounded_fork_dag,
    "feature_collider": generate_feature_collider_dag,
    "erdos_renyi_dag": generate_erdos_renyi_dag,
    "layered": generate_layered_dag,
}


def generate_dag(family: str, d: int, seed: int) -> nx.DiGraph:
    """Generate a DAG from a named family with given seed and dimension d."""
    if family not in FAMILY_GENERATORS:
        raise ValueError(f"Unknown family '{family}'. Must be one of {GRAPH_FAMILIES}")
    if d < 5 or d > 10:
        raise ValueError(f"d must be between 5 and 10, got {d}")

    rng = np.random.default_rng(seed)
    G = FAMILY_GENERATORS[family](d, rng)
    G.graph["family"] = family
    G.graph["d"] = d
    G.graph["seed"] = seed
    return G


def compute_graph_metrics(G: nx.DiGraph, d: int) -> dict[str, Any]:
    """Compute structural graph properties: confounding presence and mediation share."""
    features = [f"X{i}" for i in range(1, d + 1)]
    y_parents = set(G.predecessors("Y"))

    # has_confounding: any feature that is a common ancestor of Y and another Y parent,
    # or that has >=2 children among features/Y where both lead to Y
    has_confounding = False
    for f in features:
        descendants = nx.descendants(G, f)
        # count how many distinct paths lead to Y from f
        children = list(G.successors(f))
        if len(children) >= 2:
            paths_to_y = 0
            for child in children:
                if child == "Y" or nx.has_path(G, child, "Y"):
                    paths_to_y += 1
            if paths_to_y >= 2:
                has_confounding = True
                break

    # mediation_share: fraction of features having an indirect path to Y
    # (i.e. path of length >= 2 through another feature)
    mediating_features = 0
    for f in features:
        # find all simple paths from f to Y
        paths = list(nx.all_simple_paths(G, source=f, target="Y"))
        has_indirect = any(len(p) > 2 for p in paths)
        if has_indirect:
            mediating_features += 1

    mediation_share = float(mediating_features / d)

    return {
        "has_confounding": bool(has_confounding),
        "mediation_share": float(mediation_share),
        "n_y_parents": len(y_parents),
        "total_edges": G.number_of_edges(),
    }
