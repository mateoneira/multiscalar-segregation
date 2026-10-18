"""Step 05: hierarchical tree.

Builds the cluster tree from the percolation memberships. The tree is
census-year independent — the hierarchy is fixed; only the population changes
between census years. Census-unit labels on the 2022 sector geography are
computed in step 05b.
"""

import sys

import networkx as nx
import pandas as pd

from multiscalar_segregation import load_config, repo_root, tree


def main() -> None:
    cfg = load_config()
    root = repo_root()
    interim = root / cfg["paths"]["data_interim"]
    out = interim / "tree"
    out.mkdir(parents=True, exist_ok=True)

    mem = pd.read_parquet(interim / "percolation" / "memberships.parquet")
    memberships = {
        t: g.set_index("node_id")["cluster"]
        for t, g in mem.groupby("threshold")
    }
    G = tree.build_tree(memberships, min_cluster_nodes=int(cfg["percolation"]["min_cluster_nodes"]))
    n_levels = len(memberships)
    print(f"tree: {G.number_of_nodes()} nodes, {len(tree.leaves(G))} leaves, "
          f"{n_levels} percolation levels", flush=True)
    nx.write_graphml(G, out / "tree.graphml")
    print(f"wrote {out / 'tree.graphml'}")


if __name__ == "__main__":
    sys.exit(main())
