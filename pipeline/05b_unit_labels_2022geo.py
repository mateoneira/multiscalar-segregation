"""Step 05b: census-unit labels on the 2022 sector geography.

Spatial-joins the street nodes to the 2022 census sectors (INEC Geodatabase
Nacional 2021, layer sec_a) and computes per-level cluster labels. These
labels serve BOTH census years. The 2010 microdata is harmonized to 2022
geography, so the fixed hierarchy sees identical units in both years.
"""

import sys

import networkx as nx
import pandas as pd
import pyogrio

from multiscalar_segregation import assignment, load_config, repo_root

GPKG = "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"


def main() -> None:
    cfg = load_config()
    root = repo_root()
    interim = root / cfg["paths"]["data_interim"]

    print("reading 2022 sector polygons...", flush=True)
    sectors = pyogrio.read_dataframe(root / GPKG, layer="sec_a", columns=["sec"])
    sectors = sectors.set_index("sec")
    sectors.index = sectors.index.astype(str).str.zfill(12)
    print(f"sectors: {len(sectors):,}", flush=True)

    nodes = pd.read_csv(root / cfg["network"]["nodes"], header=None,
                        names=["node_id", "lon", "lat"])
    points = assignment.nodes_geoframe(nodes["node_id"].values, nodes["lon"].values,
                                       nodes["lat"].values, crs=cfg["network"]["crs"])
    node_to_unit = assignment.join_nodes_to_units(points, sectors)
    node_to_unit.to_parquet(interim / "tree" / "node_to_unit_sec2022.parquet", index=False)
    print(f"street nodes inside a 2022 sector: {len(node_to_unit):,} / {len(nodes):,}", flush=True)

    G = nx.read_graphml(interim / "tree" / "tree.graphml")
    mem = pd.read_parquet(interim / "percolation" / "memberships.parquet")
    memberships = {t: g.set_index("node_id")["cluster"] for t, g in mem.groupby("threshold")}

    labels = assignment.compute_unit_labels(G, memberships, node_to_unit)
    labels.to_parquet(interim / "tree" / "unit_labels_sec2022.parquet", index=False)
    for level, g in labels.groupby("level"):
        print(f"level {level}: {g.label.nunique()} clusters, {len(g):,} units labelled")


if __name__ == "__main__":
    sys.exit(main())
