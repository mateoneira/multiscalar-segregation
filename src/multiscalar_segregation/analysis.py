"""GJSD over the hierarchical tree and characteristic segregation scales.

- Census units are assigned to a cluster per level by majority vote of their
  street nodes (see assignment.py), considering retained clusters only.
- A parent's children are its retained sub-clusters; units that belong to the
  parent but to none of the retained children are pooled into a single
  *remainder* pseudo-child.
- Weights are shares of dwellings (households).
- Parents whose split is trivial (a single non-empty group) carry no GJSD.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from . import metrics, tree as tree_mod


def unit_cluster_labels(membership: pd.Series, level: int) -> pd.Series:
    """Tree-node labels ('level-cluster') for a per-unit cluster membership."""
    return membership.map(lambda c: tree_mod.node_name(level, c))


def gjsd_table(
    G: nx.DiGraph,
    unit_labels: dict[int, pd.Series],
    unit_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Normalised GJSD and entropy index for every parent node of the tree.

    Parameters
    ----------
    unit_labels : level -> Series(unit_id -> tree-node label at that level),
        restricted to retained clusters. The root level (max level) must map
        every unit to ``tree.ROOT``.
    unit_counts : DataFrame indexed by unit_id; one column per group; values
        are dwelling counts per group in the unit.
    """
    n_groups = unit_counts.shape[1]
    log_k = np.log(n_groups)
    records = []
    for parent in G.nodes:
        level = G.nodes[parent]["level"]
        if level not in unit_labels:
            continue
        parent_units = unit_labels[level].index[unit_labels[level] == parent]
        if len(parent_units) == 0:
            continue
        parent_counts = unit_counts.reindex(parent_units).fillna(0.0)

        children = list(G.successors(parent))
        groups: list[np.ndarray] = []
        covered = pd.Index([])
        if children and (level - 1) in unit_labels:
            child_labels = unit_labels[level - 1]
            for child in children:
                child_units = child_labels.index[child_labels == child]
                covered = covered.union(child_units)
                vec = unit_counts.reindex(child_units).fillna(0.0).to_numpy().sum(axis=0)
                if vec.sum() > 0:
                    groups.append(vec)
        remainder_units = parent_units.difference(covered)
        remainder = unit_counts.reindex(remainder_units).fillna(0.0).to_numpy().sum(axis=0)
        if remainder.sum() > 0:
            groups.append(remainder)

        parent_vec = parent_counts.to_numpy().sum(axis=0)
        total = parent_vec.sum()
        if total == 0:
            continue
        h_parent = metrics.entropy(parent_vec / total)

        if len(groups) > 1:
            weights = np.array([g.sum() for g in groups])
            weights = weights / weights.sum()
            dists = np.stack([g / g.sum() for g in groups])
            raw = metrics.gjsd(dists, weights)
            upper = metrics.gjsd_max(weights)
            norm = raw / upper if upper > 0 else 0.0
        else:
            raw = upper = norm = np.nan

        records.append({
            "node": parent,
            "level": level,
            "threshold": G.nodes[parent].get("threshold", np.inf),
            "n_children": len(children),
            "n_groups_split": len(groups),
            "n_units": int(len(parent_units)),
            "dwellings": float(total),
            "remainder_dwellings": float(remainder.sum()),
            "gjsd": raw,
            "gjsd_max": upper,
            "norm_gjsd": norm,
            "h_index": (log_k - h_parent) / log_k,
        })
    return pd.DataFrame.from_records(records).set_index("node")


def level_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Median and mean normalised GJSD per tree level."""
    valid = table.dropna(subset=["norm_gjsd"])
    return valid.groupby("level").agg(
        threshold=("threshold", "first"),
        n_parents=("norm_gjsd", "size"),
        median_norm_gjsd=("norm_gjsd", "median"),
        mean_norm_gjsd=("norm_gjsd", "mean"),
        median_h_index=("h_index", "median"),
    )


def upsilon(G: nx.DiGraph, table: pd.DataFrame) -> pd.DataFrame:
    """Characteristic segregation scale per leaf: the ancestor with maximal
    normalised GJSD along the path from the leaf to the root."""
    records = []
    for leaf in tree_mod.leaves(G):
        path = tree_mod.ancestry(G, leaf)[1:]
        scored = table.reindex([n for n in path if n in table.index]).dropna(subset=["norm_gjsd"])
        if scored.empty:
            continue
        best = scored["norm_gjsd"].idxmax()
        records.append({
            "leaf": leaf,
            "leaf_level": G.nodes[leaf]["level"],
            "upsilon_node": best,
            "upsilon_level": int(scored.loc[best, "level"]),
            "upsilon_threshold": float(scored.loc[best, "threshold"]),
            "upsilon_norm_gjsd": float(scored.loc[best, "norm_gjsd"]),
        })
    return pd.DataFrame.from_records(records).set_index("leaf")
