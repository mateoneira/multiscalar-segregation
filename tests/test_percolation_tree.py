import networkx as nx
import numpy as np
import pandas as pd
import pytest

from multiscalar_segregation import percolation, tree

EDGES = pd.DataFrame(
    [(0, 1, 10), (1, 2, 10), (0, 2, 10),
     (3, 4, 20), (4, 5, 20), (3, 5, 20),
     (2, 3, 100), (6, 7, 500)],
    columns=["source", "target", "length"],
)
NODES = np.arange(8)


def partition(labels: np.ndarray) -> set[frozenset]:
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, set()).add(i)
    return {frozenset(g) for g in groups.values()}


def test_percolation_memberships():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[15, 50, 100, 500], curve_step=5)
    assert partition(res.memberships[15]) == {
        frozenset({0, 1, 2}), frozenset({3}), frozenset({4}), frozenset({5}),
        frozenset({6}), frozenset({7})}
    assert partition(res.memberships[50]) == {
        frozenset({0, 1, 2}), frozenset({3, 4, 5}), frozenset({6}), frozenset({7})}
    assert partition(res.memberships[100]) == {
        frozenset({0, 1, 2, 3, 4, 5}), frozenset({6}), frozenset({7})}
    assert partition(res.memberships[500]) == {
        frozenset({0, 1, 2, 3, 4, 5}), frozenset({6, 7})}


def test_percolation_curve():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[500], curve_step=5)
    curve = res.curve.set_index("threshold")
    assert curve.loc[10, "largest_component"] == 3
    assert curve.loc[100, "largest_component"] == 6
    assert curve.loc[100, "n_components"] == 3
    assert curve.loc[500, "n_components"] == 2


def test_percolation_transition_diagnostics_and_merges():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[500], curve_step=5,
                                merge_min_size=3)
    curve = res.curve.set_index("threshold")
    assert curve.loc[20, "second_component"] == 3
    assert curve.loc[95, "second_component"] == 3
    assert curve.loc[100, "second_component"] == 1
    assert curve.loc[10, "susceptibility"] == pytest.approx(5 / 8)
    assert curve.loc[20, "susceptibility"] == pytest.approx(11 / 8)
    assert curve.loc[100, "susceptibility"] == pytest.approx(2 / 8)
    assert curve.loc[500, "susceptibility"] == pytest.approx(4 / 8)
    assert curve.loc[curve["susceptibility"].idxmax(), "susceptibility"] == \
        curve.loc[20, "susceptibility"]
    assert res.merges.values.tolist() == [[100.0, 3, 3]]
    assert len(res.merges) == 1


def test_select_thresholds_flags_jumps():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[500], curve_step=5)
    jumps = tree.select_thresholds(res.curve, min_jump=0.2)
    assert set(jumps["threshold"]) == {10.0, 100.0}
    jumps_strict = tree.select_thresholds(res.curve, min_jump=0.3)
    assert set(jumps_strict["threshold"]) == {100.0}


def test_build_tree_structure_and_cutoff():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[50, 100, 500], curve_step=5)
    memberships = {t: res.membership_series(t) for t in (50, 100, 500)}
    G = tree.build_tree(memberships, min_cluster_nodes=3)

    lv = nx.get_node_attributes(G, "level")
    assert sorted(n for n, l in lv.items() if l == 0) == sorted(
        [tree.node_name(0, memberships[50].loc[0]), tree.node_name(0, memberships[50].loc[3])])
    merged = tree.node_name(1, memberships[100].loc[0])
    assert G.out_degree(merged) == 2
    assert set(G.successors(tree.ROOT)) == {tree.node_name(2, memberships[500].loc[0])}
    for leaf in tree.leaves(G):
        assert tree.ancestry(G, leaf)[-1] == tree.ROOT


def test_gjsd_over_tree_with_remainder_semantics():
    from multiscalar_segregation import analysis
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[50, 100, 500], curve_step=5)
    memberships = {t: res.membership_series(t) for t in (50, 100, 500)}
    G = tree.build_tree(memberships, min_cluster_nodes=3)

    unit_labels = {}
    for level, t in enumerate((50, 100, 500)):
        m = memberships[t]
        retained_names = {n.split("-", 1)[1] for n, d in G.nodes(data=True)
                          if n != tree.ROOT and d["level"] == level}
        m = m[m.astype(str).isin(retained_names)]
        unit_labels[level] = analysis.unit_cluster_labels(m, level)
    unit_labels[3] = pd.Series(tree.ROOT, index=NODES)

    unit_counts = pd.DataFrame(
        {"g1": [1, 1, 1, 0, 0, 0, 1, 0], "g2": [0, 0, 0, 1, 1, 1, 0, 1]},
        index=NODES, dtype=float)
    table = analysis.gjsd_table(G, unit_labels, unit_counts)

    merged = tree.node_name(1, memberships[100].loc[0])
    top = tree.node_name(2, memberships[500].loc[0])
    assert table.loc[merged, "norm_gjsd"] == pytest.approx(1.0)
    assert np.isnan(table.loc[top, "norm_gjsd"])
    assert table.loc[tree.ROOT, "norm_gjsd"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc[tree.ROOT, "remainder_dwellings"] == 2.0

    ups = analysis.upsilon(G, table)
    for leaf in ups.index:
        assert ups.loc[leaf, "upsilon_node"] == merged
