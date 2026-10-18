"""Assign census units to percolation clusters via the street nodes they contain.

Each census unit (sector/manzana polygon) receives, per hierarchy level, the
cluster label held by the majority of the street-network nodes falling inside
it.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def nodes_geoframe(node_ids: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                   crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"node_id": node_ids},
        geometry=gpd.points_from_xy(lon, lat),
        crs=crs,
    )


def join_nodes_to_units(nodes: gpd.GeoDataFrame, units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join: street nodes -> containing census unit (index of ``units``)."""
    right = units.rename_axis("unit_id").to_crs(nodes.crs)
    joined = gpd.sjoin(nodes, right[["geometry"]], how="inner", predicate="within")
    return joined[["node_id", "unit_id"]]


def majority_membership(
    node_to_unit: pd.DataFrame,
    membership: pd.Series,
) -> pd.Series:
    df = node_to_unit.copy()
    df["cluster"] = membership.reindex(df["node_id"].values).values
    df = df.dropna(subset=["cluster"])
    # reindex over missing nodes upcasts to float; restore the source dtype
    df["cluster"] = df["cluster"].astype(membership.dtype)
    counts = df.groupby(["unit_id", "cluster"]).size().rename("n").reset_index()
    top = counts.sort_values("n", kind="stable").drop_duplicates("unit_id", keep="last")
    return top.set_index("unit_id")["cluster"]


def compute_unit_labels(G, memberships: dict[float, pd.Series],
                        node_to_unit: pd.DataFrame) -> pd.DataFrame:
    """Per-level census-unit labels ('level-cluster') for retained clusters.

    Returns a long DataFrame (unit_id, label, level).
    """
    from . import analysis, tree as tree_mod

    frames = []
    for level, t in enumerate(sorted(memberships)):
        retained = {n.split("-", 1)[1] for n, d in G.nodes(data=True)
                    if n != tree_mod.ROOT and d["level"] == level}
        m = memberships[t]
        m = m[m.astype(str).isin(retained)]
        maj = majority_membership(node_to_unit, m)
        lab = analysis.unit_cluster_labels(maj, level).rename("label")
        frames.append(lab.reset_index().assign(level=level))
    return pd.concat(frames, ignore_index=True)


def assignment_report(units: pd.DataFrame, assigned: pd.Series,
                      population_col: str = "total_population") -> dict:
    """Population accounting for units without a cluster assignment."""
    total = float(units[population_col].sum())
    covered = float(units.loc[units.index.isin(assigned.index), population_col].sum())
    return {
        "n_units": int(len(units)),
        "n_assigned": int(len(assigned)),
        "population_total": total,
        "population_assigned": covered,
        "population_unassigned_share": 1.0 - covered / total if total else np.nan,
    }
