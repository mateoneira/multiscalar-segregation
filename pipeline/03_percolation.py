"""Step 03: percolation on the canonical street network.

Writes per-level membership tables and the largest-component curve.
"""

import sys

import numpy as np
import pandas as pd

from multiscalar_segregation import load_config, percolation, repo_root


def main() -> None:
    cfg = load_config()
    root = repo_root()
    out = root / cfg["paths"]["data_interim"] / "percolation"
    out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(root / cfg["network"]["nodes"], header=None,
                        names=["node_id", "lon", "lat"])
    edges, edge_universe = percolation.load_edgelist(str(root / cfg["network"]["edgelist"]))
    universe = np.union1d(nodes["node_id"].values, edge_universe)
    print(f"nodes: {len(universe):,}  edges: {len(edges):,}", flush=True)

    levels = [float(t) for t in cfg["percolation"]["levels_m"]]
    res = percolation.percolate(
        edges, universe,
        snapshot_thresholds=levels,
        curve_step=float(cfg["percolation"]["sweep_step_m"]),
        curve_max=float(cfg["percolation"]["sweep_max_m"]),
    )

    frames = [res.membership_series(t).rename("cluster").reset_index()
                 .rename(columns={"index": "node_id"}).assign(threshold=t)
              for t in levels]
    memberships = pd.concat(frames, ignore_index=True)
    memberships.to_parquet(out / "memberships.parquet", index=False)

    processed = root / cfg["paths"]["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    res.curve.to_csv(processed / "percolation_curve.csv", index=False)
    print(f"wrote {out / 'memberships.parquet'} and percolation_curve.csv")


if __name__ == "__main__":
    sys.exit(main())
