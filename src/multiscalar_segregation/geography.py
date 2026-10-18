from __future__ import annotations

import pandas as pd

# digits of the INEC dwelling id ID_VIV (2022 geography); the first 12 are the sector
ID_VIV_DIGITS = {"2010": 18, "2022": 19}

REGIONS = ("Costa", "Sierra", "Amazonía")

PROVINCE_REGION = {
    "07": "Costa",   "08": "Costa",   "09": "Costa",   "12": "Costa",
    "13": "Costa",   "23": "Costa",   "24": "Costa",
    "01": "Sierra",  "02": "Sierra",  "03": "Sierra",  "04": "Sierra",
    "05": "Sierra",  "06": "Sierra",  "10": "Sierra",  "11": "Sierra",
    "17": "Sierra",  "18": "Sierra",
    "14": "Amazonía", "15": "Amazonía", "16": "Amazonía",
    "19": "Amazonía", "21": "Amazonía", "22": "Amazonía",
}


def node_regions(node_to_unit: pd.DataFrame) -> pd.Series:
    """node_id -> region, from the province prefix of the unit's sector code."""
    prov = node_to_unit["unit_id"].astype(str).str.zfill(12).str[:2]
    return pd.Series(prov.map(PROVINCE_REGION).values,
                     index=node_to_unit["node_id"], name="region")


def cluster_regions(memberships: pd.DataFrame, regions: pd.Series,
                    node_name) -> tuple[pd.Series, pd.Series]:
    """Majority region of every percolation cluster.

    ``memberships`` holds (node_id, cluster, threshold); levels are the sort
    order of the thresholds, matching ``tree.build_tree``. Returns two Series
    indexed by tree node name: the majority region and the share of the
    cluster's regionally-attributed nodes that fall in it.
    """
    m = memberships.assign(region=memberships["node_id"].map(regions))
    m = m.dropna(subset=["region"])
    levels = {t: i for i, t in enumerate(sorted(memberships["threshold"].unique()))}
    m["node"] = [node_name(levels[t], c)
                 for t, c in zip(m["threshold"], m["cluster"])]

    counts = m.groupby(["node", "region"]).size().unstack(fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    return share.idxmax(axis=1), share.max(axis=1)
