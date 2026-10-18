"""Hierarchical tree of percolation clusters across retained thresholds."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

ROOT = "root"


def node_name(level: int, cluster) -> str:
    return f"{level}-{cluster}"


def select_thresholds(curve: pd.DataFrame, min_jump: float) -> pd.DataFrame:
    """Candidate hierarchy levels from jumps in the largest-component curve.

    A threshold is retained when the relative size of the largest connected
    component increases by more than ``min_jump`` in one grid step. Returns the
    jump table (threshold, jump) sorted by threshold.
    """
    jumps = curve["largest_relative"].diff()
    selected = curve.loc[jumps > min_jump, ["threshold"]].copy()
    selected["jump"] = jumps[jumps > min_jump].values
    return selected.reset_index(drop=True)


def build_tree(
    memberships: dict[float, pd.Series],
    min_cluster_nodes: int = 600,
) -> nx.DiGraph:
    """Build the cluster hierarchy from per-threshold node memberships.

    Parameters
    ----------
    memberships : threshold -> Series(node_id -> cluster label). Thresholds are
        sorted ascending; their sort order defines tree levels 0..L-1.
    min_cluster_nodes : clusters with fewer street-network nodes are dropped
        (they are absorbed by an ancestor at the level where they exceed the
        cutoff, or never enter the tree).

    Returns a DiGraph with edges parent -> child, node attributes ``level``
    (int; the synthetic root is L), ``threshold`` (m) and ``n_nodes`` (cluster
    size in street-network nodes). Because percolation components only merge as
    the threshold grows, each retained cluster has exactly one parent: the
    cluster containing any of its member nodes at the next level up.
    """
    thresholds = sorted(memberships)
    G = nx.DiGraph()

    retained: dict[int, pd.Series] = {}
    for level, delta in enumerate(thresholds):
        m = memberships[delta]
        sizes = m.value_counts()
        keep = sizes[sizes >= min_cluster_nodes]
        retained[level] = m[m.isin(keep.index)]
        for cluster, size in keep.items():
            G.add_node(node_name(level, cluster), level=level,
                       threshold=float(delta), n_nodes=int(size))

    for level in range(len(thresholds) - 1):
        child_m = retained[level]
        parent_m = memberships[thresholds[level + 1]]
        if child_m.empty:
            continue
        reps = child_m.groupby(child_m).apply(lambda s: s.index[0])
        for cluster, rep_node in reps.items():
            parent_cluster = parent_m.loc[rep_node]
            parent = node_name(level + 1, parent_cluster)
            if parent in G:
                G.add_edge(parent, node_name(level, cluster))

    top = len(thresholds) - 1
    top_nodes = [n for n, d in G.nodes(data=True) if d["level"] == top]
    G.add_node(ROOT, level=top + 1, threshold=np.inf,
               n_nodes=int(sum(G.nodes[n]["n_nodes"] for n in top_nodes)))
    for n in top_nodes:
        G.add_edge(ROOT, n)
    return G


def leaves(G: nx.DiGraph) -> list[str]:
    return [n for n in G.nodes if G.out_degree(n) == 0]


def ancestry(G: nx.DiGraph, leaf: str) -> list[str]:
    """Path from a leaf up to the root (leaf first)."""
    path = [leaf]
    while True:
        preds = list(G.predecessors(path[-1]))
        if not preds:
            return path
        path.append(preds[0])
