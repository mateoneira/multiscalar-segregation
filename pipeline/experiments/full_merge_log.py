"""Experiment: exact macroscopic merge events over the FULL edge-length range.

The diagnostics sweep (detect_transitions.py) records merge events only up to
sweep_max_m (12,000 m), but the manuscript quotes the last Amazon merges at
17-20 km and the synthetic root's span up to 19,519 m. This script runs the
same grid-free union-find merge log with no distance cap (up to the longest
edge, ~52 km) so those numbers are reproducible from a committed artifact.

Writes data/interim/experiments/merge_events_full.csv and prints every
macroscopic merge (both clusters >= min_cluster_nodes) beyond the top retained
hierarchy level, i.e. the events collapsed into the synthetic root.

Usage: python pipeline/experiments/full_merge_log.py [--recompute]
"""

import sys

import numpy as np
import pandas as pd

from multiscalar_segregation import load_config, percolation, repo_root

CFG = load_config()
ROOT = repo_root()
CACHE = ROOT / CFG["paths"]["data_interim"] / "experiments"

MIN_CLUSTER = int(CFG["percolation"]["min_cluster_nodes"])
TOP_LEVEL = max(float(t) for t in CFG["percolation"]["levels_m"])


def main(recompute: bool = False) -> None:
    out_path = CACHE / "merge_events_full.csv"
    if out_path.exists() and not recompute:
        print(f"using cached {out_path}", flush=True)
        merges = pd.read_csv(out_path)
    else:
        nodes = pd.read_csv(ROOT / CFG["network"]["nodes"], header=None,
                            names=["node_id", "lon", "lat"])
        edges, edge_universe = percolation.load_edgelist(str(ROOT / CFG["network"]["edgelist"]))
        universe = np.union1d(nodes["node_id"].values, edge_universe)
        print(f"nodes: {len(universe):,}  edges: {len(edges):,}  "
              f"max edge: {edges['length'].max():,.0f} m", flush=True)
        res = percolation.percolate(edges, universe, snapshot_thresholds=[],
                                    curve_step=1000.0, curve_max=None,
                                    merge_min_size=MIN_CLUSTER)
        CACHE.mkdir(parents=True, exist_ok=True)
        merges = res.merges
        merges.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(merges)} events)", flush=True)

    late = merges[merges["length"] > TOP_LEVEL].sort_values("length")
    print(f"\nmacroscopic merges (both clusters >= {MIN_CLUSTER} nodes) "
          f"beyond the top retained level ({TOP_LEVEL:,.0f} m):")
    print(late.to_string(index=False,
                         formatters={"length": "{:,.0f}".format}))
    print(f"\nroot span: {late['length'].min():,.0f} - {late['length'].max():,.0f} m")


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv[1:])
