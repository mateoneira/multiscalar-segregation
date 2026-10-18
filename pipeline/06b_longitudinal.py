"""Step 06b: 2010 vs 2022 comparison on the fixed hierarchy.

Consumes the node tables and Upsilon tables produced by 06_gjsd.py for the two
census years (suffixes '2010' and '2022', both on 2022 sector geography) and
writes the longitudinal outputs: per-node deltas, per-level profile shifts, and
the per-city characteristic-scale (Upsilon) transition table.
"""

import sys

import pandas as pd

from multiscalar_segregation import load_config, repo_root


def main() -> None:
    cfg = load_config()
    processed = repo_root() / cfg["paths"]["data_processed"]

    nodes = {}
    ups = {}
    for year in ("2010", "2022"):
        nodes[year] = pd.read_csv(processed / f"gjsd_nodes_{year}.csv", index_col=0)
        ups[year] = pd.read_csv(processed / f"upsilon_{year}.csv", index_col=0)

    j = nodes["2010"][["level", "threshold", "norm_gjsd", "dwellings"]].join(
        nodes["2022"][["norm_gjsd", "dwellings"]], how="inner",
        lsuffix="_2010", rsuffix="_2022")
    j["delta_norm_gjsd"] = j["norm_gjsd_2022"] - j["norm_gjsd_2010"]
    j.to_csv(processed / "gjsd_nodes_delta.csv")

    prof = j.dropna(subset=["delta_norm_gjsd"]).groupby("level").agg(
        threshold=("threshold", "first"),
        n=("delta_norm_gjsd", "size"),
        median_2010=("norm_gjsd_2010", "median"),
        median_2022=("norm_gjsd_2022", "median"),
        median_delta=("delta_norm_gjsd", "median"),
    )
    prof.to_csv(processed / "gjsd_levels_delta.csv")
    print("level profile (median norm GJSD):")
    print(prof.to_string())

    u = ups["2010"][["upsilon_threshold", "upsilon_norm_gjsd"]].join(
        ups["2022"][["upsilon_threshold", "upsilon_norm_gjsd"]], how="inner",
        lsuffix="_2010", rsuffix="_2022")
    u["scale_changed"] = u["upsilon_threshold_2010"] != u["upsilon_threshold_2022"]
    u.to_csv(processed / "upsilon_transitions.csv")
    moved = u[u["scale_changed"]]
    print(f"\ncities whose characteristic scale changed: {len(moved)} / {len(u)}")
    trans = (u.groupby(["upsilon_threshold_2010", "upsilon_threshold_2022"])
             .size().rename("n").reset_index())
    trans.to_csv(processed / "upsilon_transition_matrix.csv", index=False)
    print(trans.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
