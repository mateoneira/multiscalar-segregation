import numpy as np
import pandas as pd

from multiscalar_segregation import analysis, robustness, tree
from tests.test_percolation_tree import EDGES, NODES
from multiscalar_segregation import percolation


def test_engine_matches_gjsd_table():
    res = percolation.percolate(EDGES, NODES, snapshot_thresholds=[50, 100, 500],
                                curve_step=5)
    memberships = {t: res.membership_series(t) for t in (50, 100, 500)}
    G = tree.build_tree(memberships, min_cluster_nodes=3)
    unit_labels = {}
    for level, t in enumerate((50, 100, 500)):
        m = memberships[t]
        retained = {n.split("-", 1)[1] for n, d in G.nodes(data=True)
                    if n != tree.ROOT and d["level"] == level}
        m = m[m.astype(str).isin(retained)]
        unit_labels[level] = analysis.unit_cluster_labels(m, level)
    unit_labels[3] = pd.Series(tree.ROOT, index=NODES)
    unit_counts = pd.DataFrame(
        {"g1": [1, 1, 1, 0, 0, 0, 1, 0], "g2": [0, 0, 0, 1, 1, 1, 0, 1]},
        index=NODES, dtype=float)

    table = analysis.gjsd_table(G, unit_labels, unit_counts)
    engine = robustness.GjsdEngine(G, unit_labels, unit_counts.index)
    fast = engine.norm_gjsd(unit_counts.to_numpy())

    joined = table["norm_gjsd"].to_frame("slow").join(fast.rename("fast"), how="outer")
    assert ((joined["slow"] - joined["fast"]).abs().fillna(0) < 1e-12).all()
    assert joined["slow"].isna().equals(joined["fast"].isna())


def test_permuted_unit_counts_preserves_margins():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.integers(0, 20, (30, 10)).astype(float),
                          index=[f"u{i}" for i in range(30)],
                          columns=[f"g{i}" for i in range(10)])
    perm = robustness.permuted_unit_counts(counts, rng)
    np.testing.assert_array_equal(perm.sum(axis=1).values, counts.sum(axis=1).values)
    np.testing.assert_array_equal(perm.sum(axis=0).values, counts.sum(axis=0).values)
    assert not np.array_equal(perm.values, counts.values)


def test_regroup_matches_direct_quantile_binning():
    rng = np.random.default_rng(1)
    icv = pd.Series(rng.normal(size=5000))
    units = pd.Series(rng.integers(0, 40, 5000)).map(lambda i: f"u{i}")
    for n in (4, 5, 10):
        counts = robustness.regroup_unit_counts(icv, units, n)
        assert counts.shape[1] == n
        assert counts.to_numpy().sum() == 5000
        margins = counts.sum(axis=0).to_numpy()
        assert margins.max() - margins.min() <= n
