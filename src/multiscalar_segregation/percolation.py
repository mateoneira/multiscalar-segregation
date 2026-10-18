"""Distance-threshold percolation on a weighted street network.

For a threshold delta, keep all
edges with length <= delta and label the connected components. A single
union-find sweep over the length-sorted edge list yields memberships at any set
of thresholds plus the largest-component curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class _UnionFind:
    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> tuple[int, int, int]:
        """Union the sets of a and b.

        Returns (merged_size, larger_pre_size, smaller_pre_size); the smaller
        pre-merge size is 0 when a and b were already in the same set.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            s = int(self.size[ra])
            return s, s, 0
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        sa, sb = int(self.size[ra]), int(self.size[rb])
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return sa + sb, sa, sb


@dataclass
class PercolationResult:
    node_ids: np.ndarray          # original node identifiers, index-aligned with memberships
    thresholds: np.ndarray        # requested snapshot thresholds (sorted)
    memberships: dict[float, np.ndarray]  # threshold -> component root per node
    curve: pd.DataFrame           # per-threshold diagnostics (see percolate docstring)
    merges: pd.DataFrame | None = None    # macroscopic merge events (length, size_large, size_small)

    def membership_series(self, threshold: float) -> pd.Series:
        """Component label per node id at a snapshot threshold."""
        return pd.Series(self.memberships[threshold], index=self.node_ids, name="cluster")

    def cluster_sizes(self, threshold: float) -> pd.Series:
        return self.membership_series(threshold).value_counts()


def load_edgelist(path: str, node_ids: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Read a headerless source,target,length_m edge list; return edges + nodes."""
    edges = pd.read_csv(path, header=None, names=["source", "target", "length"],
                        dtype={"source": np.int64, "target": np.int64, "length": float})
    if node_ids is None:
        node_ids = pd.unique(pd.concat([edges["source"], edges["target"]]).values)
        node_ids.sort()
    return edges, np.asarray(node_ids)


def percolate(
    edges: pd.DataFrame,
    node_ids: np.ndarray,
    snapshot_thresholds: list[float],
    curve_step: float = 10.0,
    curve_max: float | None = None,
    merge_min_size: int | None = None,
) -> PercolationResult:
    """Sweep thresholds in one pass and snapshot memberships.

    Component labels in each snapshot are union-find roots (arbitrary but
    consistent integers).

    The curve carries the finite-size transition diagnostics at each grid
    threshold: largest_component (s1), second_component (s2), n_components,
    and susceptibility chi = sum_{i>=2} s_i^2 / N  (Stauffer & Aharony).

    If merge_min_size is given, every union of two components that are BOTH
    at least that many nodes is recorded in ``merges`` (exact edge length and
    the two pre-merge sizes).
    """
    idx = pd.Series(np.arange(len(node_ids)), index=node_ids)
    src = idx.loc[edges["source"].values].values
    dst = idx.loc[edges["target"].values].values
    lengths = edges["length"].values

    order = np.argsort(lengths, kind="stable")
    src, dst, lengths = src[order], dst[order], lengths[order]

    uf = _UnionFind(len(node_ids))
    snapshots = sorted(snapshot_thresholds)
    memberships: dict[float, np.ndarray] = {}

    curve_max = curve_max if curve_max is not None else float(lengths[-1])
    curve_grid = np.arange(curve_step, curve_max + curve_step, curve_step)
    curve_rows = []

    n = len(node_ids)
    node_range = np.arange(n)
    merge_rows: list[tuple[float, int, int]] = []
    e = 0
    n_edges = len(lengths)
    snap_iter = iter(snapshots)
    next_snap = next(snap_iter, None)
    for delta in curve_grid:
        while e < n_edges and lengths[e] <= delta:
            _, big, small = uf.union(src[e], dst[e])
            if merge_min_size is not None and small >= merge_min_size:
                merge_rows.append((float(lengths[e]), big, small))
            e += 1
        while next_snap is not None and next_snap <= delta:
            memberships[next_snap] = np.array([uf.find(i) for i in range(n)])
            next_snap = next(snap_iter, None)
        sizes = uf.size[uf.parent == node_range]
        n_components = int(sizes.size)
        s1 = int(sizes.max())
        s2 = int(np.partition(sizes, -2)[-2]) if n_components > 1 else 0
        chi = float(np.dot(sizes, sizes) - s1 * s1) / n
        curve_rows.append((float(delta), s1, n_components, s2, chi))

    while next_snap is not None:
        while e < n_edges and lengths[e] <= next_snap:
            _, big, small = uf.union(src[e], dst[e])
            if merge_min_size is not None and small >= merge_min_size:
                merge_rows.append((float(lengths[e]), big, small))
            e += 1
        memberships[next_snap] = np.array([uf.find(i) for i in range(n)])
        next_snap = next(snap_iter, None)

    curve = pd.DataFrame(curve_rows, columns=[
        "threshold", "largest_component", "n_components", "second_component", "susceptibility"])
    curve["largest_relative"] = curve["largest_component"] / n
    curve["second_relative"] = curve["second_component"] / n
    merges = pd.DataFrame(merge_rows, columns=["length", "size_large", "size_small"])
    return PercolationResult(np.asarray(node_ids), np.array(snapshots), memberships, curve, merges)
