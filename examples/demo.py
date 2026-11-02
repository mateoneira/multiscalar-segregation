"""Demo: percolation divergence trees on a small synthetic urban system.

Builds a toy street network of eight grid towns grouped into two regions,
assigns each street node a synthetic population split into five
socio-economic groups, and runs the full method: percolation, hierarchical
tree, normalised GJSD per parent node, and the characteristic scale of each
town. Needs no external data; runs in a few seconds.

    uv run python examples/demo.py
"""

import numpy as np
import pandas as pd

from multiscalar_segregation import analysis, percolation, tree

RNG = np.random.default_rng(42)
SPACING = 50.0          # street segment length within a town (m)
TOWN_SIDE = 10          # each town is a 10 x 10 grid of intersections
THRESHOLDS = [60.0, 500.0, 2500.0]
N_GROUPS = 5


def grid_town(offset: int, x0: float, y0: float):
    """Nodes and edges of one square grid town."""
    ids = offset + np.arange(TOWN_SIDE * TOWN_SIDE)
    xy = np.array([(x0 + i * SPACING, y0 + j * SPACING)
                   for i in range(TOWN_SIDE) for j in range(TOWN_SIDE)])
    edges = []
    for i in range(TOWN_SIDE):
        for j in range(TOWN_SIDE):
            u = offset + i * TOWN_SIDE + j
            if i + 1 < TOWN_SIDE:
                edges.append((u, u + TOWN_SIDE, SPACING))
            if j + 1 < TOWN_SIDE:
                edges.append((u, u + 1, SPACING))
    return ids, xy, edges


def build_network():
    """Eight towns: pairs 300 m apart, pairs 1.5 km apart within a region,
    and the two regions 4 km apart."""
    origins = [(0, 0), (750, 0), (0, 1950), (750, 1950),
               (8000, 0), (8750, 0), (8000, 1950), (8750, 1950)]
    ids, xy, edges, town = [], [], [], []
    for t, (x0, y0) in enumerate(origins):
        i, p, e = grid_town(t * TOWN_SIDE ** 2, x0, y0)
        ids.append(i); xy.append(p); edges += e; town += [t] * len(i)
    ids, xy, town = np.concatenate(ids), np.vstack(xy), np.array(town)

    def link(a, b):
        """Shortest straight link between towns a and b."""
        pa, pb = np.where(town == a)[0], np.where(town == b)[0]
        d = np.linalg.norm(xy[pa][:, None] - xy[pb][None], axis=2)
        i, j = np.unravel_index(d.argmin(), d.shape)
        edges.append((ids[pa[i]], ids[pb[j]], float(d[i, j])))

    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (4, 6), (1, 4)]:
        link(a, b)
    return pd.DataFrame(edges, columns=["source", "target", "length"]), ids, town


def synthetic_population(town: np.ndarray) -> pd.DataFrame:
    """Dwelling counts per group for every street node. Each town has its own
    mean group, so towns differ within pairs, pairs differ within regions, and
    region 2 (towns 4-7) is poorer overall."""
    town_shift = [0.5, 2.5, 1.0, 1.2, 2.6, 2.8, 2.0, 4.0]
    counts = np.zeros((len(town), N_GROUPS))
    for k, t in enumerate(town):
        shift = town_shift[t]
        p = np.exp(-0.8 * (np.arange(N_GROUPS) - shift) ** 2)
        counts[k] = RNG.multinomial(40, p / p.sum())
    return pd.DataFrame(counts, columns=[f"g{g}" for g in range(N_GROUPS)])


def main():
    edges, node_ids, town = build_network()
    unit_counts = synthetic_population(town).set_index(node_ids)

    res = percolation.percolate(edges, node_ids, snapshot_thresholds=THRESHOLDS,
                                curve_step=10)
    memberships = {t: res.membership_series(t) for t in THRESHOLDS}
    G = tree.build_tree(memberships, min_cluster_nodes=50)

    # here every street node is its own census unit
    unit_labels = {}
    for level, t in enumerate(THRESHOLDS):
        retained = {n.split("-", 1)[1] for n, d in G.nodes(data=True)
                    if n != tree.ROOT and d["level"] == level}
        m = memberships[t]
        unit_labels[level] = analysis.unit_cluster_labels(m[m.astype(str).isin(retained)], level)
    unit_labels[len(THRESHOLDS)] = pd.Series(tree.ROOT, index=node_ids)

    table = analysis.gjsd_table(G, unit_labels, unit_counts)
    print(f"{G.number_of_nodes()} tree nodes, {len(node_ids)} street nodes, "
          f"{int(unit_counts.to_numpy().sum())} dwellings\n")
    print("Normalised GJSD per level:")
    print(analysis.level_summary(table).round(3).to_string(), "\n")
    print("Characteristic scale (Upsilon) per town:")
    ups = analysis.upsilon(G, table)
    print(ups[["upsilon_threshold", "upsilon_norm_gjsd"]].round(3).to_string())


if __name__ == "__main__":
    main()
