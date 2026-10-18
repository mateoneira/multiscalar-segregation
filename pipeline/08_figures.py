"""Step 08: publication figures (main + SI), all scripted.

Usage: 08_figures.py [fig ...]   (no args = all available)

Figures are written to figures/ as vector PDF + 600 dpi PNG with a shared
style (multiscalar_segregation.viz).
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from multiscalar_segregation import geography, load_config, repo_root, tree, viz

CFG = load_config()
ROOT = repo_root()
PROCESSED = ROOT / CFG["paths"]["data_processed"]
INTERIM = ROOT / CFG["paths"]["data_interim"]
FIGS = ROOT / CFG["paths"]["figures"]
YEARS = ("2010", "2022")
LEVELS_M = [int(t) for t in CFG["percolation"]["levels_m"]]
LEVEL_LABELS = {t: viz.level_label(t) for t in LEVELS_M}
PROFILE_LEVELS = LEVELS_M[1:]
SIMPLIFY_M = 100
TREE_TICKS = list(LEVEL_LABELS.values()) + ["national"]


def nodes_lonlat():
    return pd.read_csv(ROOT / CFG["network"]["nodes"], header=None,
                       names=["node_id", "lon", "lat"]).set_index("node_id")


_COUNTRY = None


def country_geometry():
    """Cached national outline (mainland, dissolved census sectors)."""
    global _COUNTRY
    if _COUNTRY is None:
        import geopandas as gpd
        path = INTERIM / "ecuador_outline.geojson"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run pipeline/02_country_outline.py")
        _COUNTRY = gpd.read_file(path).geometry.iloc[0]
    return _COUNTRY


def draw_country(ax, fill=True):
    """Land fill + national boundary beneath the data layers (zorder <= 0.5)."""
    geom = country_geometry()
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        xs, ys = p.exterior.xy
        if fill:
            ax.fill(xs, ys, color=viz.LAND, linewidth=0, zorder=0)
        ax.plot(xs, ys, color=viz.BASELINE, lw=0.5, zorder=0.5,
                solid_capstyle="round")


def map_frame(ax):
    """Shared frame for whole-Ecuador maps: fixed extent, no chrome."""
    ax.set_xlim(-81.2, -75.2)
    ax.set_ylim(-5.1, 1.6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


LEVEL_CAPTURES = {
    0: "city cores",
    1: "cities and conurbations",
    2: "metropolitan regions",
    3: "regional sub-systems of cities",
    4: "coast, highlands and Amazon split",
    5: "unified coastal system",
    6: "coastal system and northern highlands",
    7: "Country (root)",
}
CRITICAL_LEVELS = tuple(i for i, t in enumerate(LEVELS_M)
                        if t in CFG["percolation"]["critical_m"])

KEY_CITIES = [("Guayaquil", "0-41239"), ("Quito", "0-124263"),
              ("Cuenca", "0-286115"), ("Machala", "0-309462"),
              ("Manta", "0-35915")]

PEAK_NAMES = {"2-139491": "Guaranda–Guanujo"}

CITY_RENAME = {
    "SANTO DOMINGO DE LOS COLORADOS": "Santo Domingo",
    "DURAN": "Durán",
}

_CLUSTER_LON = None
_TREE_POS = None


def canton_seats():
    """Canton code -> name of its seat, i.e. the city (from the census geodatabase)."""
    import pyogrio
    gpkg = ("data/raw/geodatabase/GEODATABASE_NACIONAL_2021/"
            "GEODATABASE_NACIONAL_2021.gpkg")
    a = pyogrio.read_dataframe(ROOT / gpkg, layer="aream_a", read_geometry=False)
    kind = a["tipo_aream"].astype(str).str.upper()
    # "CABERECA" is a typo in the INEC source data
    seats = a[kind.str.contains("CANTONAL|CABERECA|CAPITAL", na=False)].copy()
    seats["canton"] = seats["parroquia"].astype(str).str.zfill(6).str[:4]
    return seats.drop_duplicates("canton").set_index("canton")["nom_aman"]


def leaf_cities(G, n=12):
    """The n largest leaf clusters, named after the seat of the canton holding most
    of their dwellings"""
    seats = canton_seats()
    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")
    counts = pd.read_parquet(INTERIM / "units_2022" / "unit_counts.parquet")["num_dwellings"]

    rows = []
    for leaf in tree.leaves(G):
        units = labels[(labels["level"] == G.nodes[leaf]["level"])
                       & (labels["label"] == leaf)]["unit_id"]
        dw = counts.reindex(units).dropna()
        if dw.empty:
            continue
        by_canton = dw.groupby(pd.Series(dw.index.str[:4], index=dw.index)).sum()
        city = seats.get(by_canton.idxmax())
        if city is None:
            continue
        city = CITY_RENAME.get(city, city.title())
        rows.append({"leaf": leaf, "city": city, "dwellings": float(dw.sum())})

    d = pd.DataFrame(rows).sort_values("dwellings", ascending=False)
    return d.drop_duplicates("city").head(n)


def cluster_lon():
    """Median longitude of every percolation cluster, keyed by tree node name."""
    global _CLUSTER_LON
    if _CLUSTER_LON is None:
        mem = pd.read_parquet(INTERIM / "percolation" / "memberships.parquet")
        coords = nodes_lonlat()
        _CLUSTER_LON = {}
        for level, t in enumerate(sorted(mem["threshold"].unique())):
            m = mem[mem["threshold"] == t].set_index("node_id")["cluster"]
            med = coords.reindex(m.index).assign(cluster=m.values) \
                        .groupby("cluster")["lon"].median()
            for cl, v in med.items():
                _CLUSTER_LON[tree.node_name(level, cl)] = v
    return _CLUSTER_LON


def tree_layout(G):
    """Shared layout for tree figures."""
    global _TREE_POS
    if _TREE_POS is None:
        lon = cluster_lon()
        _TREE_POS = viz.tree_positions(G, tree.ROOT,
                                       sort_key=lambda n: lon.get(n, np.inf))
    return _TREE_POS


def leaf_geography():
    """Centroid (mainland) and dwelling counts per leaf city, per year."""
    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    mem = pd.read_parquet(INTERIM / "percolation" / "memberships.parquet")
    coords = nodes_lonlat()
    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")
    thresholds = sorted(mem["threshold"].unique())

    rows = []
    for leaf in tree.leaves(G):
        level = G.nodes[leaf]["level"]
        cluster = leaf.split("-", 1)[1]
        t = thresholds[level]
        m = mem[mem["threshold"] == t]
        ids = m.loc[m["cluster"].astype(str) == cluster, "node_id"]
        c = coords.reindex(ids)[["lon", "lat"]].median()
        rows.append({"leaf": leaf, "lon": c["lon"], "lat": c["lat"],
                     "n_nodes": G.nodes[leaf]["n_nodes"]})
    geo = pd.DataFrame(rows).set_index("leaf")

    for year in YEARS:
        counts = pd.read_parquet(INTERIM / f"units_{year}" / "unit_counts.parquet")
        for leaf in geo.index:
            level = G.nodes[leaf]["level"]
            units = labels[(labels["level"] == level) & (labels["label"] == leaf)]
            geo.loc[leaf, f"dwellings_{year}"] = \
                counts["num_dwellings"].reindex(units["unit_id"]).sum()
    return G, geo


def fig_percolation_curve():
    curve = pd.read_csv(PROCESSED / "percolation_curve.csv")
    fig, ax = plt.subplots(figsize=(viz.SINGLE_COL, 2.2))
    ax.plot(curve["threshold"], curve["largest_relative"], color=viz.CAT[0],
            lw=1.4, solid_capstyle="round")
    for i, t in enumerate(CFG["percolation"]["levels_m"]):
        ax.axvline(t, color=viz.BASELINE, lw=0.5, zorder=0)
        ax.annotate(LEVEL_LABELS[t], xy=(t, 1.0), xycoords="data",
                    xytext=(0, 2 + 7 * (i % 2)), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=viz.MUTED,
                    annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(100, 12000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("percolation threshold δ (m)")
    ax.set_ylabel("largest component\n(share of nodes)")
    viz.save(fig, FIGS, "percolation_curve")


def fig_level_profile_delta():
    prof = pd.read_csv(PROCESSED / "gjsd_levels_delta.csv", index_col=0)
    prof = prof[np.isfinite(prof["threshold"])]
    x = np.arange(len(prof))
    fig, ax = plt.subplots(figsize=(viz.SINGLE_COL, 2.4))
    for xi, (_, r) in zip(x, prof.iterrows()):
        ax.plot([xi, xi], [r["median_2010"], r["median_2022"]],
                color=viz.BASELINE, lw=1.0, zorder=1)
    for year, col in (("2010", "median_2010"), ("2022", "median_2022")):
        ax.scatter(x, prof[col], s=22, color=viz.YEAR_COLORS[year], zorder=3,
                   label=year, edgecolor=viz.SURFACE, linewidth=0.6)
    for xi, (_, r) in zip(x, prof.iterrows()):
        pct = 100 * r["median_delta"] / r["median_2010"]
        ax.annotate(f"{pct:.0f}%".replace("-", "−"),
                    xy=(xi, min(r["median_2010"], r["median_2022"])),
                    xytext=(0, -11), textcoords="offset points", ha="center",
                    fontsize=6, color=viz.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS.get(int(t), "national")
                        for t in prof["threshold"]])
    ax.set_ylabel("median normalised GJSD")
    ax.set_xlabel("hierarchy level (percolation threshold)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", handletextpad=0.4)
    viz.save(fig, FIGS, "level_profile_delta")


def fig_upsilon_map():
    G, geo = leaf_geography()
    coords = nodes_lonlat()
    bg = coords.sample(120_000, random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(viz.DOUBLE_COL, 3.4), sharey=True)
    for ax, year in zip(axes, YEARS):
        ups = pd.read_csv(PROCESSED / f"upsilon_{year}.csv", index_col=0)
        d = geo.join(ups, how="inner")
        cls = d["upsilon_threshold"].map(lambda t: viz.scale_class(t, LEVELS_M))
        draw_country(ax)
        ax.scatter(bg["lon"], bg["lat"], s=0.05, color=viz.BASELINE,
                   rasterized=True, linewidths=0, zorder=1)
        size = 6 + 40 * np.sqrt(d[f"dwellings_{year}"] / d[f"dwellings_{year}"].max())
        ax.scatter(d["lon"], d["lat"], s=size,
                   c=[viz.ORDINAL_5[int(c)] for c in cls],
                   edgecolor=viz.SURFACE, linewidth=0.5, zorder=2)
        import matplotlib.patheffects as pe
        for city, label in [("0-124263", "Quito"), ("0-41239", "Guayaquil"),
                            ("0-286115", "Cuenca")]:
            if city in d.index:
                ax.annotate(label, (d.loc[city, "lon"] + 0.18, d.loc[city, "lat"]),
                            fontsize=6.5, color=viz.INK_2, va="center", zorder=6,
                            path_effects=[pe.withStroke(linewidth=1.6,
                                                        foreground=viz.SURFACE)])
        viz.panel_label(ax, "a" if year == "2010" else "b", year)
        map_frame(ax)
    viz.scale_bar(axes[0])
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=5,
                          markerfacecolor=c, markeredgecolor=viz.SURFACE,
                          label=l)
               for c, l in zip(viz.ORDINAL_5, viz.scale_class_labels(LEVELS_M))]
    fig.legend(handles=handles, title="characteristic scale of segregation, Υ",
               loc="lower center", ncol=5, fontsize=6.5, title_fontsize=6.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.02)
    viz.save(fig, FIGS, "upsilon_map")


def fig_gjsd_profiles():
    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    fig, axes = plt.subplots(1, 2, figsize=(viz.DOUBLE_COL, 2.6), sharey=True)
    for ax, year in zip(axes, YEARS):
        table = pd.read_csv(PROCESSED / f"gjsd_nodes_{year}.csv", index_col=0)
        ups = pd.read_csv(PROCESSED / f"upsilon_{year}.csv", index_col=0)
        xpos = {t: i for i, t in enumerate(sorted(LEVEL_LABELS) + [np.inf])}
        for leaf in tree.leaves(G):
            path = tree.ancestry(G, leaf)[1:]
            scored = table.reindex(path).dropna(subset=["norm_gjsd"])
            if scored.empty:
                continue
            xs = [xpos[t] for t in scored["threshold"]]
            ax.plot(xs, scored["norm_gjsd"], color=viz.MUTED, lw=0.4, alpha=0.35,
                    zorder=1)
        ax.scatter([xpos[t] for t in ups["upsilon_threshold"]],
                   ups["upsilon_norm_gjsd"], s=9, color=viz.MUTED,
                   alpha=0.7, linewidths=0, zorder=3, label="Υ (per city)")
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=1.6, foreground=viz.SURFACE)]
        starts = []
        for slot, (name, leaf) in enumerate(KEY_CITIES):
            c = viz.CAT[slot]
            path = tree.ancestry(G, leaf)[1:]
            scored = table.reindex(path).dropna(subset=["norm_gjsd"])
            xs = [xpos[t] for t in scored["threshold"]]
            ys = scored["norm_gjsd"].tolist()
            ax.plot(xs, ys, color=c, lw=0.9, zorder=2)
            u = ups.loc[leaf]
            ax.scatter([xpos[u["upsilon_threshold"]]], [u["upsilon_norm_gjsd"]],
                       s=16, color=c, edgecolor=viz.SURFACE, linewidth=0.3,
                       zorder=4)
            starts.append((name, xs[0], ys[0], c))
        starts.sort(key=lambda r: r[2])
        min_gap = 0.022
        y_lab: list[float] = []
        for _, _, y, _ in starts:
            y_lab.append(y if not y_lab else max(y, y_lab[-1] + min_gap))
        for (name, x, y, c), yl in zip(starts, y_lab):
            ax.annotate(name, (x - 0.18, yl), ha="right", va="center",
                        fontsize=5.5, color=c, zorder=6, path_effects=halo)
            ax.plot([x - 0.15, x], [yl, y], color=c, lw=0.5, alpha=0.7,
                    zorder=2)
        at_108 = table[table["threshold"] == 1080]
        peak = at_108["norm_gjsd"].idxmax()
        ax.annotate(PEAK_NAMES.get(peak, peak),
                    (xpos[1080] + 0.12, at_108.loc[peak, "norm_gjsd"]),
                    ha="left", va="center", fontsize=5.5, color=viz.CAT[5],
                    zorder=6, path_effects=halo)
        med = table.dropna(subset=["norm_gjsd"]).groupby("threshold")["norm_gjsd"].median()
        ax.plot([xpos[t] for t in med.index], med.values, color=viz.INK, lw=1.2,
                ls="--", zorder=4, label="median")
        ticks = PROFILE_LEVELS + [np.inf]
        ax.set_xticks([xpos[t] for t in ticks])
        ax.set_xticklabels([LEVEL_LABELS.get(t, "nat.") for t in ticks],
                           fontsize=6)
        ax.set_xlim(xpos[340] - 1.15, xpos[np.inf] + 0.5)
        viz.panel_label(ax, "a" if year == "2010" else "b", year)
        ax.set_xlabel("percolation threshold δ")
    axes[0].set_ylabel("normalised GJSD")
    axes[0].legend(loc="upper right")
    viz.save(fig, FIGS, "gjsd_profiles")


def fig_city_profiles():
    """Main figure: GJSD profiles of the five largest cities (leaf clusters by
    household count; same top 5 in both censuses), 2010 vs 2022."""
    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    tables = {y: pd.read_csv(PROCESSED / f"gjsd_nodes_{y}.csv", index_col=0)
              for y in YEARS}
    ups = {y: pd.read_csv(PROCESSED / f"upsilon_{y}.csv", index_col=0) for y in YEARS}
    cities = KEY_CITIES
    xpos = {t: i for i, t in enumerate(sorted(LEVEL_LABELS) + [np.inf])}
    short = {340: "0.34", 1080: "1.1", 1960: "2", 4070: "4.1", 5950: "6", 6340: "6.3"}
    fig, axes = plt.subplots(1, 5, figsize=(viz.DOUBLE_COL, 2.0), sharey=True)
    for ax, letter, (name, leaf) in zip(axes, "abcde", cities):
        path = tree.ancestry(G, leaf)[1:]
        for year in YEARS:
            scored = tables[year].reindex(path).dropna(subset=["norm_gjsd"])
            xs = [xpos[t] for t in scored["threshold"]]
            ax.plot(xs, scored["norm_gjsd"], color=viz.YEAR_COLORS[year], lw=1.2,
                    label=year, solid_capstyle="round")
            u = ups[year].loc[leaf]
            ax.scatter([xpos[u["upsilon_threshold"]]], [u["upsilon_norm_gjsd"]],
                       s=26, facecolor=viz.SURFACE, zorder=5,
                       edgecolor=viz.YEAR_COLORS[year], linewidth=1.2)
        ticks = PROFILE_LEVELS + [np.inf]
        ax.set_xticks([xpos[t] for t in ticks])
        ax.set_xticklabels([short.get(t, "nat.") for t in ticks], fontsize=6)
        ax.set_xlim(xpos[340] - 0.4, xpos[np.inf] + 0.4)
        viz.panel_label(ax, letter, name)
        ax.set_xlabel("δ (km)")
    axes[0].set_ylabel("normalised GJSD")
    axes[0].legend(loc="upper right", handlelength=1.2, handletextpad=0.5,
                   borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.08)
    viz.save(fig, FIGS, "city_profiles")


GPKG = "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"


def leaf_names(G, geo):
    """Place name per leaf city, from the INEC geodatabase name layers.

    Each leaf is named after its (dwelling-weighted) modal parroquia:
    parroquia cabecera / provincial capital first, then any amanzanada
    locality of the parroquia, then the locality of the modal sector, then
    the canton cabecera. When two leaves resolve to the same name (a city
    split by the street network, e.g. Guayaquil south of the estuaries),
    the smaller one gets a suffix relative to the larger.
    """
    import pyogrio

    am = pyogrio.read_dataframe(ROOT / GPKG, layer="aream_a", read_geometry=False)
    am["parroquia"] = am["parroquia"].astype(str).str.zfill(6)
    am["canton"] = am["parroquia"].str[:4]
    prio = {"CAPITAL PROVINCIAL": 0, "CABECERA CANTONAL": 0, "CABERECA CANTONAL": 0,
            "CABECERA PARROQUIAL": 1}
    am["prio"] = am["tipo_aream"].map(prio).fillna(2)
    am = am.sort_values("prio")
    canton = am[am["prio"] == 0].drop_duplicates("canton").set_index("canton")["nom_aman"]
    par_head = am[am["prio"] < 2].drop_duplicates("parroquia").set_index("parroquia")["nom_aman"]
    par_any = am.drop_duplicates("parroquia").set_index("parroquia")["nom_aman"]

    loc = pyogrio.read_dataframe(ROOT / GPKG, layer="loc_p", read_geometry=False)
    loc["sector"] = loc["loc"].astype(str).str.zfill(15).str[:12]
    sec_loc = loc.drop_duplicates("sector").set_index("sector")["n_loc"]

    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")
    counts = pd.read_parquet(INTERIM / "units_2022" / "unit_counts.parquet")

    def pretty(s):
        t = str(s).title()
        for w in (" De ", " Del ", " La ", " Los ", " Las ", " Y "):
            t = t.replace(w, w.lower())
        return t.replace("Puerto Francisco de Orellana", "Coca")

    names, weight = {}, {}
    for leaf in tree.leaves(G):
        level = G.nodes[leaf]["level"]
        units = labels[(labels["level"] == level) & (labels["label"] == leaf)]
        sectors = units["unit_id"].astype(str).str.zfill(12)
        dw = counts["num_dwellings"].reindex(units["unit_id"]).fillna(0).values
        by_par = pd.Series(dw, index=sectors.str[:6]).groupby(level=0).sum()
        par = by_par.idxmax()
        modal_sector = sectors.iloc[int(np.argmax(dw))]
        names[leaf] = pretty(par_head.get(par, par_any.get(
            par, sec_loc.get(modal_sector, canton.get(par[:4], leaf)))))
        weight[leaf] = dw.sum()
    out = pd.Series(names, name="name")
    w = pd.Series(weight)
    for name, group in out.groupby(out):
        if len(group) > 1:
            anchor = w[group.index].idxmax()
            for leaf in group.index.drop(anchor):
                dlon = geo.loc[leaf, "lon"] - geo.loc[anchor, "lon"]
                dlat = geo.loc[leaf, "lat"] - geo.loc[anchor, "lat"]
                compass = (("E" if dlon > 0 else "W") if abs(dlon) > abs(dlat)
                           else ("N" if dlat > 0 else "S"))
                out[leaf] = f"{name} ({compass})"
    return out


def fig_change_map():
    """Main figure: where segregation deepened, 2010-2022. (a) Change in each
    city's peak (characteristic-scale) normalised GJSD on the map; (b) 2010 vs
    2022 peak values against the identity line; (c) the largest changes."""
    import matplotlib.patheffects as pe

    G, geo = leaf_geography()
    tr = pd.read_csv(PROCESSED / "upsilon_transitions.csv", index_col=0)
    d = geo.join(tr, how="inner").dropna(subset=["upsilon_norm_gjsd_2010",
                                                 "upsilon_norm_gjsd_2022"])
    d["delta"] = d["upsilon_norm_gjsd_2022"] - d["upsilon_norm_gjsd_2010"]
    coords = nodes_lonlat()
    bg = coords.sample(120_000, random_state=0)
    vmax = 0.09
    halo = [pe.withStroke(linewidth=1.6, foreground=viz.SURFACE)]

    fig, (axm, axs, axr) = plt.subplots(1, 3, figsize=(viz.DOUBLE_COL, 3.6),
                                        width_ratios=[1.45, 1, 0.75])
    cities = [("0-124263", "Quito", 1), ("0-41239", "Guayaquil", 1),
              ("0-286115", "Cuenca", 1), ("1-139491", "Guaranda", -1),
              ("0-97046", "Esmeraldas", 1), ("0-109454", "Daule", -1)]
    draw_country(axm)
    axm.scatter(bg["lon"], bg["lat"], s=0.05, color=viz.BASELINE,
                rasterized=True, linewidths=0, zorder=1)
    order = np.argsort(np.abs(d["delta"].values))
    size = 6 + 40 * np.sqrt(d["dwellings_2022"] / d["dwellings_2022"].max())
    smap = axm.scatter(d["lon"].values[order], d["lat"].values[order],
                       s=size.values[order], c=d["delta"].values[order],
                       cmap=viz.DIVERGING, vmin=-vmax, vmax=vmax,
                       edgecolor=viz.SURFACE, linewidth=0.5, zorder=2)
    for city, label, side in cities:
        if city in d.index:
            axm.annotate(label, (d.loc[city, "lon"] + 0.18 * side, d.loc[city, "lat"]),
                         fontsize=6.5, color=viz.INK_2, va="center", zorder=6,
                         ha="left" if side > 0 else "right", path_effects=halo)
    viz.panel_label(axm, "a")
    map_frame(axm)
    viz.scale_bar(axm)
    cbar = fig.colorbar(smap, ax=axm, orientation="horizontal", shrink=0.52,
                        pad=0.02, aspect=26)
    cbar.set_label("Δ peak normalised GJSD, 2022 − 2010", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    lim = 1.05 * float(d[["upsilon_norm_gjsd_2010", "upsilon_norm_gjsd_2022"]]
                       .max().max())
    axs.plot([0, lim], [0, lim], color=viz.BASELINE, lw=0.6, zorder=1)
    axs.scatter(d["upsilon_norm_gjsd_2010"], d["upsilon_norm_gjsd_2022"],
                s=4 + 24 * np.sqrt(d["dwellings_2022"] / d["dwellings_2022"].max()),
                c=d["delta"], cmap=viz.DIVERGING, vmin=-vmax, vmax=vmax,
                edgecolor=viz.SURFACE, linewidth=0.4, zorder=2)
    named = {"0-124263": ("Quito", -0.008, 0, "right"),
             "0-41239": ("Guayaquil", 0.008, 0.004, "left"),
             "0-286115": ("Cuenca", 0.008, -0.006, "left"),
             "1-139491": ("Guaranda", -0.008, 0, "right"),
             "0-97046": ("Esmeraldas", -0.006, 0.014, "right"),
             "0-109454": ("Daule", -0.006, 0.012, "right")}
    for city, (label, dx, dy, ha) in named.items():
        if city in d.index:
            axs.annotate(label, (d.loc[city, "upsilon_norm_gjsd_2010"] + dx,
                                 d.loc[city, "upsilon_norm_gjsd_2022"] + dy),
                         fontsize=6, color=viz.INK_2, va="center", ha=ha,
                         zorder=6, path_effects=halo)
    axs.annotate("deepened", (0.04, lim - 0.02), fontsize=6.5, color=viz.MUTED,
                 va="top")
    axs.annotate("receded", (lim - 0.02, 0.04), fontsize=6.5, color=viz.MUTED,
                 ha="right")
    axs.set_xlim(0, lim)
    axs.set_ylim(0, lim)
    axs.set_aspect("equal")
    axs.set_xlabel("peak normalised GJSD, 2010")
    axs.set_ylabel("peak normalised GJSD, 2022")

    peak_node = {y: pd.read_csv(PROCESSED / f"upsilon_{y}.csv", index_col=0)
                 ["upsilon_node"].rename(f"node_{y}") for y in YEARS}
    r = d.join(peak_node["2010"]).join(peak_node["2022"]).join(leaf_names(G, geo))
    r = (r.sort_values("dwellings_2022", ascending=False)
          .drop_duplicates(["node_2010", "node_2022"]))
    r = pd.concat([r.nsmallest(5, "delta"), r.nlargest(5, "delta")])
    r = r.sort_values("delta")
    y = np.arange(len(r))
    shade = viz.DIVERGING(plt.Normalize(-vmax, vmax)(np.clip(r["delta"], -vmax, vmax)))
    axr.barh(y, r["delta"], color=shade, height=0.72)
    axr.axvline(0, color=viz.BASELINE, lw=0.6, zorder=0)
    axr.set_yticks(y)
    axr.set_yticklabels(r["name"], fontsize=6)
    axr.set_ylim(-0.6, len(r) - 0.4)
    axr.tick_params(axis="y", length=0)
    axr.grid(axis="y", visible=False)
    axr.set_xlabel("Δ peak normalised GJSD")
    axr.set_box_aspect(1.35)

    axm.set_anchor("N")
    fig.subplots_adjust(wspace=0.46)
    for ax in (axm, axs, axr):
        ax.apply_aspect()  # resolve drawn positions before reading them
    mbox = axm.get_position()
    for ax in (axs, axr):
        b = ax.get_position()
        ax.set_position([b.x0, (mbox.y0 + mbox.y1 - b.height) / 2,
                         b.width, b.height])
    pad = 2 / 72 / fig.get_size_inches()[1]
    for ax, letter in ((axs, "b"), (axr, "c")):
        fig.text(ax.get_position().x0, mbox.y1 + pad, letter, ha="left",
                 va="bottom", fontsize=8.5, fontweight="bold", color=viz.INK)
    viz.save(fig, FIGS, "change_map")


def fig_level_change_map():
    """Main figure: where segregation changed between the censuses, by scale.
    Δ normalised GJSD (2022 − 2010) of every node scored in both years at the
    340 m (a), 4.07 km (b) and 6.34 km (c) levels, projected onto the census
    sectors belonging to each cluster and drawn as dissolved areas on one
    shared diverging scale. Panels (d-f) zoom the 340 m level on Guayaquil, Cuenca and Riobamba."""
    import pyogrio
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle
    from shapely import from_wkt

    SHOW = [(340, "340 m — local"), (4070, "4.07 km — regional"),
            (6340, "6.34 km — macro-regions")]
    level_of = {t: i for i, t in enumerate(sorted(LEVEL_LABELS))}

    tables = {y: pd.read_csv(PROCESSED / f"gjsd_nodes_{y}.csv", index_col=0)
              for y in YEARS}
    delta = (tables["2022"]["norm_gjsd"]
             - tables["2010"]["norm_gjsd"]).dropna().rename("delta")

    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")
    counts = pd.read_parquet(INTERIM / "units_2022"
                             / "unit_counts.parquet")["num_dwellings"]
    sectors = pyogrio.read_dataframe(ROOT / GPKG, layer="sec_a", columns=["sec"])
    sectors["sec"] = sectors["sec"].astype(str).str.zfill(12)
    c = sectors.geometry.centroid.to_crs("EPSG:4326")
    cent = pd.DataFrame({"sec": sectors["sec"], "lon": c.x, "lat": c.y}) \
        .set_index("sec")

    foot = {}
    for thr, _ in SHOW:
        lab = labels[labels["level"] == level_of[thr]] \
            .set_index("unit_id")["label"]
        nodes = delta.index.intersection(lab.unique())
        if nodes.empty:
            raise ValueError(f"no scored nodes at the {thr} m level")
        g = sectors.assign(label=sectors["sec"].map(lab))
        g = g[g["label"].isin(nodes)]
        dis = g.dissolve("label")
        dis["geometry"] = dis.geometry.simplify(SIMPLIFY_M)
        foot[thr] = dis.to_crs("EPSG:4326").join(delta)
        share = counts.reindex(g["sec"]).sum() / counts.sum()
        d = foot[thr]["delta"]
        print(f"  {thr} m: {len(d)} clusters, {share:.0%} of dwellings, "
              f"Δ min/median/max = {d.min():+.3f}/{d.median():+.3f}/"
              f"{d.max():+.3f}", flush=True)

    all_d = pd.concat([foot[t]["delta"] for t, _ in SHOW])
    vmax = float(np.ceil(np.quantile(np.abs(all_d), 0.95) * 100) / 100)
    print(f"  colour scale saturates at ±{vmax:g}", flush=True)
    norm = Normalize(-vmax, vmax)

    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    coords = nodes_lonlat()
    # commas inside the WKT LINESTRINGs rule out a plain csv parse
    raw = (ROOT / "data/network/ec_edges.csv").read_text().splitlines()
    parts = pd.Series(raw).str.split(", ", n=2, expand=True)
    edges_all = pd.DataFrame({"source": parts[0].astype("int64"),
                              "target": parts[1].astype("int64"),
                              "wkt": parts[2].str.strip()})
    bg = coords.sample(120_000, random_state=0)
    halo = [pe.withStroke(linewidth=1.6, foreground=viz.SURFACE)]

    cities = [("Guayaquil", "0-41239"), ("Cuenca", "0-286115"),
              ("Riobamba", "0-219053")]
    anc340 = {}
    for name, leaf in cities:
        anc = [n for n in tree.ancestry(G, leaf)
               if G.nodes[n]["level"] == level_of[340]]
        if anc and anc[0] in foot[340].index:
            anc340[name] = anc[0]
        else:
            print(f"  warning: no scored 340 m ancestor for {name}, "
                  "zoom panel will be empty", flush=True)

    windows = {}
    for name in anc340:
        members = labels[(labels["level"] == level_of[340])
                         & (labels["label"] == anc340[name])]["unit_id"]
        m = cent[cent.index.isin(members)]
        dist = np.hypot(m["lon"] - m["lon"].median(),
                        m["lat"] - m["lat"].median())
        m = m[dist <= 3 * dist.quantile(0.95)]
        cx = (m["lon"].max() + m["lon"].min()) / 2
        cy = (m["lat"].max() + m["lat"].min()) / 2
        span = max(m["lon"].max() - m["lon"].min(),
                   m["lat"].max() - m["lat"].min())
        half = span / 2 + min(0.12, max(0.015, 0.9 * span))
        windows[name] = (cx - half, cx + half, cy - half, cy + half)

    fig = plt.figure(figsize=(viz.DOUBLE_COL, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0],
                          hspace=0.3, wspace=0.08)

    map_axes = []
    for col, (thr, title) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, col])
        map_axes.append(ax)
        draw_country(ax)
        ax.scatter(bg["lon"], bg["lat"], s=0.05, color=viz.BASELINE,
                   rasterized=True, linewidths=0, zorder=1)
        style = (
            dict(edgecolor="face", linewidth=0.5, rasterized=True)
            if thr == 340 else dict(edgecolor=viz.SURFACE, linewidth=0.4))
        foot[thr].plot(ax=ax, column="delta", cmap=viz.DIVERGING,
                       vmin=-vmax, vmax=vmax, zorder=2, **style)
        viz.panel_label(ax, "abc"[col], title)
        map_frame(ax)
    viz.scale_bar(map_axes[0])
    for letter, (name, leaf) in zip("def", cities):
        if name in anc340:
            x0, x1, y0, y1 = windows[name]
            map_axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                            fill=False, edgecolor=viz.INK_2,
                                            linewidth=0.6, zorder=5))
            map_axes[0].annotate(letter, (x1 + 0.04, y1), fontsize=6.5,
                                 color=viz.INK_2, ha="left", va="top",
                                 zorder=6, path_effects=halo)
    sm = plt.cm.ScalarMappable(cmap=viz.DIVERGING, norm=norm)
    cbar = fig.colorbar(sm, ax=map_axes, orientation="horizontal",
                        shrink=0.35, pad=0.03, aspect=28)
    cbar.set_label("Δ normalised GJSD, 2022 − 2010", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    leg = cbar.ax.legend(
        handles=[Patch(facecolor=viz.LAND, edgecolor=viz.BASELINE,
                       linewidth=0.5)],
        labels=["undefined Δ GJSD"], loc="center left",
        bbox_to_anchor=(1.04, 0.5), fontsize=5.5, frameon=False,
        handlelength=1.2, handleheight=0.9, borderpad=0, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(viz.INK_2)

    inset_axes = []
    for col, (name, leaf) in enumerate(cities):
        ax = fig.add_subplot(gs[1, col])
        inset_axes.append(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        viz.panel_label(ax, "def"[col], name)
        if name not in anc340:
            ax.set_axis_off()
            continue
        x0, x1, y0, y1 = windows[name]
        ax.set_facecolor(viz.LAND)
        in_box = coords[coords["lon"].between(x0, x1)
                        & coords["lat"].between(y0, y1)]
        e = edges_all[edges_all["source"].isin(in_box.index)
                      & edges_all["target"].isin(in_box.index)]
        segs = [np.asarray(g.coords)
                for g in from_wkt(e["wkt"].to_numpy())]
        ax.add_collection(LineCollection(segs, colors=viz.BASELINE,
                                         linewidths=0.35, rasterized=True,
                                         zorder=1))
        around = cent[cent["lon"].between(x0, x1)
                      & cent["lat"].between(y0, y1)]
        ax.scatter(around["lon"], around["lat"], s=1.0, color=viz.GRID,
                   linewidths=0, rasterized=True, zorder=2)
        win = foot[340].cx[x0:x1, y0:y1]
        if not win.empty:
            win.plot(ax=ax, column="delta", cmap=viz.DIVERGING,
                     vmin=-vmax, vmax=vmax, alpha=0.75,
                     edgecolor=viz.SURFACE, linewidth=0.4,
                     rasterized=True, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(viz.BASELINE)
            sp.set_linewidth(0.6)
    viz.save(fig, FIGS, "level_change_map")


def fig_tree_gjsd():
    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    pos = tree_layout(G)
    fig, axes = plt.subplots(2, 1, figsize=(viz.DOUBLE_COL, 4.6), sharex=True)
    for ax, year in zip(axes, YEARS):
        table = pd.read_csv(PROCESSED / f"gjsd_nodes_{year}.csv", index_col=0)
        for a, b in G.edges:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=viz.GRID, lw=0.4, zorder=1)
        vals = table["norm_gjsd"].reindex(list(G.nodes))
        sizes = pd.Series({n: G.nodes[n]["n_nodes"] for n in G.nodes})
        s = 2 + 30 * np.sqrt(sizes / sizes.max())
        colors = [viz.GJSD_CMAP(viz.GJSD_NORM(v)) if np.isfinite(v) else viz.GRID
                  for v in vals]
        xs = [pos[n][0] for n in G.nodes]
        ys = [pos[n][1] for n in G.nodes]
        ax.scatter(xs, ys, s=[s[n] for n in G.nodes], c=colors, zorder=2,
                   edgecolor=viz.BASELINE, linewidth=0.3)
        viz.panel_label(ax, "a" if year == "2010" else "b", year)
        ax.set_yticks(range(8))
        ax.set_yticklabels(TREE_TICKS)
        ax.grid(False)
        ax.set_xticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=viz.GJSD_CMAP, norm=viz.GJSD_NORM)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, label="normalised GJSD",
                        pad=0.01, spacing="uniform", ticks=viz.GJSD_BINS)
    cbar.ax.set_yticklabels([f"{b:g}" for b in viz.GJSD_BINS])
    viz.save(fig, FIGS, "tree_gjsd")


def _cluster_map_axis(ax, coords, membership, n_top=8, point_size=0.15):
    """Scatter the street nodes coloured by their percolation cluster (top-n
    clusters in fixed categorical order."""
    top = membership.value_counts().head(n_top).index
    assert n_top <= len(viz.CAT), "one fixed slot per cluster, never cycled"
    base = coords.reindex(membership.index)
    draw_country(ax)
    ax.scatter(base["lon"], base["lat"], s=point_size * 0.4, color=viz.BASELINE,
               rasterized=True, linewidths=0, zorder=1)
    for slot, cl in enumerate(top):
        ids = membership.index[membership == cl]
        pts = coords.reindex(ids)
        ax.scatter(pts["lon"], pts["lat"], s=point_size,
                   color=viz.CAT[slot], rasterized=True,
                   linewidths=0, zorder=2)
    map_frame(ax)


def fig_percolation_maps():
    """Main figure: largest-component curve plus cluster maps at four levels."""
    curve = pd.read_csv(PROCESSED / "percolation_curve.csv")
    mem = pd.read_parquet(INTERIM / "percolation" / "memberships.parquet")
    coords = nodes_lonlat()

    fig = plt.figure(figsize=(viz.DOUBLE_COL, 4.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(curve["threshold"], curve["largest_relative"], color=viz.CAT[0],
            lw=1.4, solid_capstyle="round")
    for i, t in enumerate(CFG["percolation"]["levels_m"]):
        ax.axvline(t, color=viz.BASELINE, lw=0.5, zorder=0)
        ax.annotate(LEVEL_LABELS[t], xy=(t, 1.0), xycoords="data",
                    xytext=(0, 3 + 8 * (i % 2)), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=viz.MUTED,
                    annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(100, 12000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("percolation threshold δ (m)")
    ax.set_ylabel("largest component\n(share of nodes)")
    viz.panel_label(ax, "a", pad=10)

    for i, (letter, t) in enumerate(zip("bcde", (160, 1080, 4070, 6340))):
        axm = fig.add_subplot(gs[1, i])
        m = mem[mem["threshold"] == t].set_index("node_id")["cluster"]
        _cluster_map_axis(axm, coords, m)
        viz.panel_label(axm, letter, f"δ = {LEVEL_LABELS[t]}")
    viz.save(fig, FIGS, "percolation_maps")


def fig_hierarchical_tree():
    """Main figure: the cluster hierarchy, node size = cluster size, colour =
    the natural region (Costa / Sierra / Amazonía) holding most of the cluster's
    street nodes."""
    import matplotlib.patheffects as pe

    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    mem = pd.read_parquet(INTERIM / "percolation" / "memberships.parquet")
    n2u = pd.read_parquet(INTERIM / "tree" / "node_to_unit_sec2022.parquet")

    region, purity = geography.cluster_regions(
        mem, geography.node_regions(n2u), tree.node_name)
    pure = purity.reindex([n for n in G.nodes if n != tree.ROOT])
    print(f"  clusters >=95% one region: {(pure >= 0.95).sum()}/{len(pure)}"
          f" (min purity {pure.min():.2f})", flush=True)

    pos = tree_layout(G)
    fig, ax = plt.subplots(figsize=(viz.DOUBLE_COL, 3.3))
    fig.subplots_adjust(left=0.085, right=0.775)

    for a, b in G.edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=viz.BASELINE, lw=0.45, zorder=1, solid_capstyle="round")

    sizes = pd.Series({n: G.nodes[n]["n_nodes"] for n in G.nodes})
    n_max = sizes.max()

    def area(n_nodes):
        return 3 + 38 * np.sqrt(n_nodes / n_max)

    nodes = list(G.nodes)
    colors = [viz.MUTED if n == tree.ROOT
              else viz.REGION_COLORS[region[n]] for n in nodes]
    ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
               s=[area(sizes[n]) for n in nodes], c=colors, zorder=2,
               edgecolor=viz.SURFACE, linewidth=0.3)

    cities = leaf_cities(G)
    for leaf, label in zip(cities["leaf"], cities["city"]):
        ax.annotate(label, (pos[leaf][0], pos[leaf][1] - 0.3), ha="center",
                    va="top", rotation=90, fontsize=6, color=viz.INK_2, zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground=viz.SURFACE)])

    ax.set_ylim(-0.55, 7.4)
    ax.set_yticks(range(8))
    ax.set_yticklabels(TREE_TICKS)
    ax.set_ylabel("percolation threshold δ")
    ax.grid(False)
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    for i, tick in enumerate(ax.get_yticklabels()):
        if i in CRITICAL_LEVELS:
            tick.set_fontweight("bold")

    for level, text in LEVEL_CAPTURES.items():
        ax.annotate(text, xy=(1.015, level), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=6, color=viz.MUTED,
                    annotation_clip=False)

    entries = [(viz.REGION_COLORS[r], r) for r in geography.REGIONS]
    entries.append((viz.MUTED, "national (all regions)"))
    leg = viz.swatch_legend(ax, entries, loc="lower left",
                            bbox_to_anchor=(0.0, 1.0), ncols=4,
                            columnspacing=1.1)
    ax.add_artist(leg)
    viz.size_legend(ax, area, [(1_000, "1k"), (30_000, "30k"), (300_000, "300k")],
                    title="cluster size (network nodes)", loc="lower right",
                    bbox_to_anchor=(1.0, 1.0), ncols=3, columnspacing=1.0)
    viz.save(fig, FIGS, "hierarchical_tree")


def fig_gjsd_tree_2010():
    """Main figure: the 2010 tree coloured by norm GJSD.
    """
    import pyogrio
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")
    table = pd.read_csv(PROCESSED / "gjsd_nodes_2010.csv", index_col=0)

    sectors = pyogrio.read_dataframe(ROOT / GPKG, layer="sec_a", columns=["sec"])
    sectors["sec"] = sectors["sec"].astype(str).str.zfill(12)
    c = sectors.geometry.centroid.to_crs("EPSG:4326")
    cent = pd.DataFrame({"sec": sectors["sec"], "lon": c.x, "lat": c.y}) \
        .set_index("sec")
    counts = pd.read_parquet(INTERIM / "units_2010" / "unit_counts.parquet")
    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")
    d_icv = cent.join(counts[["mean_icv"]], how="inner").dropna()
    edges_all = pd.read_csv(ROOT / CFG["network"]["edgelist"], header=None,
                            names=["source", "target", "length_m"])
    coords_all = nodes_lonlat()

    ex = [
        ("0-222132", "1-222132", "segregated at street scale", ["1-222132"]),
        ("1-139491", "2-139491", "a divide at a single scale", ["2-139491"]),
        ("0-219053", "2-219053", "segregated at every scale",
         ["1-219053", "2-219053", "3-219053"]),
        ("0-343249", "5-343249", "segregation grows with scale", ["5-343249"]),
    ]
    city = {"0-222132": "Guayaquil (S)", "1-139491": "Guaranda–Guanujo",
            "0-219053": "Riobamba", "0-343249": "Loja"}
    acc = viz.CAT[5]

    prof_ticks = PROFILE_LEVELS
    xpos = {t: i for i, t in enumerate(prof_ticks + [np.inf])}

    fig = plt.figure(figsize=(viz.DOUBLE_COL, 6.8))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.55, 1.15, 0.95],
                          hspace=0.24, wspace=0.24)

    axt = fig.add_subplot(gs[0, :])
    pos = tree_layout(G)
    for a, b in G.edges:
        axt.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                 color=viz.GRID, lw=0.4, zorder=1)
    vals = table["norm_gjsd"].reindex(list(G.nodes))
    sizes = pd.Series({n: G.nodes[n]["n_nodes"] for n in G.nodes})
    s = 3 + 34 * np.sqrt(sizes / sizes.max())
    colors = [viz.GJSD_CMAP(viz.GJSD_NORM(v)) if np.isfinite(v) else viz.GRID
              for v in vals]
    axt.scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes],
                s=[s[n] for n in G.nodes], c=colors, zorder=2,
                edgecolor=viz.BASELINE, linewidth=0.3)
    for letter, (leaf, node, phen, marked) in zip("bcde", ex):
        axt.scatter([pos[node][0]], [pos[node][1]], s=70, facecolor="none",
                    edgecolor=acc, linewidth=1.2, zorder=3)
        axt.annotate(letter, (pos[node][0], pos[node][1]),
                     xytext=(5, 5), textcoords="offset points", ha="left",
                     fontsize=7, fontweight="bold", color=acc, zorder=6,
                     path_effects=[pe.withStroke(linewidth=1.8,
                                                 foreground=viz.SURFACE)])
    viz.panel_label(axt, "a")
    axt.set_yticks(range(8))
    axt.set_yticklabels(TREE_TICKS)
    axt.set_ylabel("percolation threshold δ")
    axt.grid(False)
    axt.set_xticks([])
    for sp in ("left", "bottom"):
        axt.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=viz.GJSD_CMAP, norm=viz.GJSD_NORM)
    cbar = fig.colorbar(sm, ax=axt, pad=0.01, shrink=0.9, aspect=28,
                        label="normalised GJSD (2010)",
                        spacing="uniform", ticks=viz.GJSD_BINS)
    cbar.ax.set_yticklabels([f"{b:g}" for b in viz.GJSD_BINS])
    cbar.ax.tick_params(labelsize=6)

    prof_axes = []
    map_axes = []
    for col, (leaf, node, phen, marked) in enumerate(ex):
        lvl = int(table.loc[node, "level"])
        members = labels[(labels["level"] == lvl)
                         & (labels["label"] == node)]["unit_id"]
        m = d_icv.loc[d_icv.index.isin(members)]

        axm = fig.add_subplot(gs[1, col])
        map_axes.append(axm)
        dist = np.hypot(m["lon"] - m["lon"].median(),
                        m["lat"] - m["lat"].median())
        m = m[dist <= 3 * dist.quantile(0.95)]
        cx = (m["lon"].max() + m["lon"].min()) / 2
        cy = (m["lat"].max() + m["lat"].min()) / 2
        span = max(m["lon"].max() - m["lon"].min(),
                   m["lat"].max() - m["lat"].min())
        half = span / 2 + min(0.12, max(0.015, 0.9 * span))
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
        axm.set_facecolor(viz.LAND)
        in_box = coords_all[(coords_all["lon"].between(x0, x1))
                            & (coords_all["lat"].between(y0, y1))]
        e = edges_all[edges_all["source"].isin(in_box.index)
                      & edges_all["target"].isin(in_box.index)]
        segs = np.stack([coords_all.loc[e["source"], ["lon", "lat"]].to_numpy(),
                         coords_all.loc[e["target"], ["lon", "lat"]].to_numpy()],
                        axis=1)
        axm.add_collection(LineCollection(segs, colors=viz.SURFACE,
                                          linewidths=0.35, rasterized=True,
                                          zorder=1))
        around = d_icv[(d_icv["lon"].between(x0, x1))
                       & (d_icv["lat"].between(y0, y1))]
        axm.scatter(around["lon"], around["lat"], s=1.6, color=viz.BASELINE,
                    linewidths=0, rasterized=True, zorder=2)
        mu = float(m["mean_icv"].mean())
        sd = float(m["mean_icv"].std())
        sc = axm.scatter(m["lon"], m["lat"], s=4,
                         c=(m["mean_icv"] - mu) / sd, cmap=viz.DIVERGING,
                         vmin=-2, vmax=2,
                         linewidths=0.2, edgecolors=viz.BASELINE,
                         rasterized=True, zorder=3)
        axm.set_xlim(x0, x1)
        axm.set_ylim(y0, y1)
        axm.set_aspect("equal")
        axm.set_anchor("N")
        axm.set_xticks([])
        axm.set_yticks([])
        axm.grid(False)
        for sp in axm.spines.values():
            sp.set_visible(True)
            sp.set_color(viz.BASELINE)
            sp.set_linewidth(0.6)
        viz.panel_label(axm, "bcde"[col])
        axm.annotate(phen, xy=(0, 1), xycoords="axes fraction",
                     xytext=(10, 2.6), textcoords="offset points",
                     ha="left", va="bottom", fontsize=6.8, color=viz.INK,
                     annotation_clip=False)
        axm.annotate(city[leaf], xy=(0.04, 0.965), xycoords="axes fraction",
                     ha="left", va="top", fontsize=6.5, color=acc,
                     path_effects=[pe.withStroke(linewidth=1.6,
                                                 foreground=viz.SURFACE)])
        axp = fig.add_subplot(gs[2, col],
                              sharey=prof_axes[0] if prof_axes else None)
        prof_axes.append(axp)
        for lf in tree.leaves(G):
            path = tree.ancestry(G, lf)[1:]
            scored = table.reindex(path).dropna(subset=["norm_gjsd"])
            if scored.empty:
                continue
            axp.plot([xpos[t] for t in scored["threshold"]],
                     scored["norm_gjsd"],
                     color=viz.MUTED, lw=0.35, alpha=0.3, zorder=1)
        path = tree.ancestry(G, leaf)[1:]
        scored = table.reindex(path).dropna(subset=["norm_gjsd"])
        axp.plot([xpos[t] for t in scored["threshold"]], scored["norm_gjsd"],
                 color=acc, lw=1.2, zorder=3, solid_capstyle="round")
        mk = table.loc[marked]
        axp.scatter([xpos[t] for t in mk["threshold"]], mk["norm_gjsd"], s=16,
                    facecolor=viz.SURFACE, edgecolor=acc, linewidth=1.0,
                    zorder=4)
        axp.set_xticks(range(len(prof_ticks) + 1))
        axp.set_xticklabels([LEVEL_LABELS[t] for t in prof_ticks] + ["nat."],
                            fontsize=5, rotation=55, ha="right")
        axp.set_xlim(-0.4, len(prof_ticks) + 0.4)
        axp.tick_params(axis="y", labelsize=6)
        if col == 0:
            axp.set_ylabel("normalised GJSD", fontsize=6.5)
        else:
            plt.setp(axp.get_yticklabels(), visible=False)

    cax = map_axes[1].inset_axes([0.42, -0.16, 1.4, 0.055])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("sector mean ICV, deviation from cluster mean (σ)",
                 fontsize=5.5, labelpad=1.6)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.ax.set_xticklabels(["≤ −2", "−1", "0", "+1", "≥ +2"], fontsize=5.5)
    cb.ax.tick_params(length=1.5, pad=1.2)
    cb.outline.set_linewidth(0.4)
    hdl = Line2D([], [], ls="none", marker="o", ms=2.6, color=viz.BASELINE)
    leg = map_axes[0].legend([hdl], ["census sectors outside the cluster"],
                             loc="upper center", bbox_to_anchor=(0.5, -0.095),
                             frameon=False, fontsize=5.5, handletextpad=0.2,
                             borderpad=0, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(viz.INK_2)

    lo, hi = prof_axes[0].get_ylim()
    prof_axes[0].set_ylim(lo, hi + 0.08 * (hi - lo))

    fig.supxlabel("percolation threshold δ", fontsize=7, y=0.012)
    viz.save(fig, FIGS, "gjsd_tree_2010")


def fig_icv_panel():
    import pyogrio

    gpkg = "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"
    sectors = pyogrio.read_dataframe(ROOT / gpkg, layer="sec_a", columns=["sec"])
    sectors["sec"] = sectors["sec"].astype(str).str.zfill(12)
    cent = sectors.geometry.centroid.to_crs("EPSG:4326")
    cent = pd.DataFrame({"sec": sectors["sec"], "lon": cent.x, "lat": cent.y}) \
        .set_index("sec")

    fig, axes = plt.subplots(2, 2, figsize=(viz.DOUBLE_COL, 5.6),
                             height_ratios=[3.2, 1])
    for col, year in enumerate(YEARS):
        counts = pd.read_parquet(INTERIM / f"units_{year}" / "unit_counts.parquet")
        d = cent.join(counts[["mean_icv", "num_dwellings"]], how="inner").dropna()
        national = np.average(d["mean_icv"], weights=d["num_dwellings"])
        dev = d["mean_icv"] - national
        vmax = 0.30
        ax = axes[0, col]
        draw_country(ax)
        order = np.argsort(np.abs(dev.values))
        ax.scatter(d["lon"].values[order], d["lat"].values[order], s=1.2,
                   c=dev.values[order], cmap=viz.DIVERGING, vmin=-vmax, vmax=vmax,
                   linewidths=0, rasterized=True, zorder=2)
        viz.panel_label(ax, "a" if col == 0 else "b", year)
        map_frame(ax)
        if col == 0:
            viz.scale_bar(ax)

        ax = axes[1, col]
        viz.panel_label(ax, "c" if col == 0 else "d")
        icv_vals = pd.read_parquet(INTERIM / f"units_{year}" / "dwellings.parquet",
                                   columns=["ICV"])["ICV"].dropna()
        edges = np.load(INTERIM / f"units_{year}" / "decile_edges.npy")
        ax.hist(icv_vals, bins=120, color=viz.SEQ_STEPS[6], linewidth=0)
        for e in edges[1:]:
            ax.axvline(e, color=viz.INK_2, lw=0.4)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("ICV (household)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=viz.DIVERGING,
                               norm=plt.Normalize(-0.30, 0.30))
    fig.colorbar(sm, ax=axes[0, :], shrink=0.5, pad=0.01,
                 label="sector mean ICV − national mean")
    viz.save(fig, FIGS, "icv_panel")


def fig_region_map():
    """SI: orientation map of the three mainland natural regions (Costa /
    Sierra / Amazonía)."""
    import pyogrio

    gpkg = "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"
    sectors = pyogrio.read_dataframe(ROOT / gpkg, layer="sec_a", columns=["sec"])
    prov = sectors["sec"].astype(str).str.zfill(12).str[:2]
    sectors["region"] = prov.map(geography.PROVINCE_REGION)
    regions = sectors.dropna(subset=["region"]).dissolve("region")
    regions["geometry"] = regions.geometry.simplify(SIMPLIFY_M)
    regions = regions.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(viz.SINGLE_COL, 3.2))
    for r in geography.REGIONS:
        regions.loc[[r]].plot(ax=ax, color=viz.REGION_COLORS[r],
                              edgecolor=viz.SURFACE, linewidth=0.4, zorder=1)
    map_frame(ax)
    viz.swatch_legend(ax, [(viz.REGION_COLORS[r], r) for r in geography.REGIONS],
                      loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3,
                      columnspacing=1.1)
    viz.scale_bar(ax)
    viz.save(fig, FIGS, "region_map")


def fig_admin_map():
    """SI: the nested administrative units of the benchmark hierarchy
    (provinces, cantons, parroquias)."""
    import pyogrio

    gpkg = "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"
    sectors = pyogrio.read_dataframe(ROOT / gpkg, layer="sec_a", columns=["sec"])
    sec = sectors["sec"].astype(str).str.zfill(12)

    fig, axes = plt.subplots(1, 3, figsize=(viz.DOUBLE_COL, 2.9))
    tiers = (("provinces", 2, 0.5), ("cantons", 4, 0.35), ("parroquias", 6, 0.25))
    for ax, letter, (name, digits, lw) in zip(axes, "abc", tiers):
        units = sectors.assign(code=sec.str[:digits]).dissolve("code")
        units["geometry"] = units.geometry.simplify(SIMPLIFY_M)
        units = units.to_crs("EPSG:4326")
        units.plot(ax=ax, color=viz.LAND, edgecolor=viz.INK_2, linewidth=lw)
        map_frame(ax)
        viz.panel_label(ax, letter, f"{len(units):,} {name}")
        if letter == "a":
            viz.scale_bar(ax)
    viz.save(fig, FIGS, "admin_map")


def fig_robustness():
    fig, axes = plt.subplots(2, 2, figsize=(viz.DOUBLE_COL, 4.6))

    ax = axes[0, 0]
    null = pd.read_csv(PROCESSED / "null_model_2010.csv", index_col=0)
    nodes = pd.read_csv(PROCESSED / "gjsd_nodes_2010.csv", index_col=0)
    d = null.join(nodes[["level", "threshold"]]).dropna(subset=["observed"])
    d = d[np.isfinite(d["threshold"])]
    rng = np.random.default_rng(0)
    x = d["level"] + rng.uniform(-0.18, 0.18, len(d))
    ax.scatter(x, d["observed"], s=5, color=viz.CAT[0], alpha=0.6, linewidths=0,
               zorder=3, label="observed")
    med = d.groupby("level")[["null_lo", "null_hi"]].median()
    ax.fill_between(med.index, med["null_lo"], med["null_hi"], color=viz.MUTED,
                    alpha=0.35, zorder=1, label="null 95% interval (median)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(sorted(d["level"].unique()))
    ax.set_xticklabels([LEVEL_LABELS[int(t)] for t in
                        sorted(d["threshold"].unique())], fontsize=6)
    ax.set_ylabel("normalised GJSD")
    viz.panel_label(ax, "a", "observed vs permutation null (2010)")
    ax.legend(loc="upper left", fontsize=6)

    ax = axes[0, 1]
    g10 = pd.read_csv(PROCESSED / "gjsd_nodes_2010_g10.csv", index_col=0)["norm_gjsd"]
    g4 = pd.read_csv(PROCESSED / "gjsd_nodes_2010_g4.csv", index_col=0)["norm_gjsd"]
    j = pd.concat([g10, g4], axis=1, keys=["deciles", "quartiles"]).dropna()
    lim = float(j.max().max()) * 1.05
    ax.plot([0, lim], [0, lim], color=viz.BASELINE, lw=0.6, zorder=1)
    ax.scatter(j["deciles"], j["quartiles"], s=6, color=viz.CAT[0], alpha=0.6,
               linewidths=0, zorder=2)
    from scipy.stats import spearmanr
    rho = spearmanr(j["deciles"], j["quartiles"]).statistic
    ax.annotate(f"Spearman ρ = {rho:.3f}", (0.05, 0.9), xycoords="axes fraction",
                fontsize=6.5, color=viz.INK_2)
    ax.set_xlabel("norm. GJSD, deciles")
    ax.set_ylabel("norm. GJSD, quartiles")
    viz.panel_label(ax, "b", "group-count sensitivity (2010)")

    for ax, year in zip(axes[1], YEARS):
        for variant, ls, label in (("minus", ":", "−200 m"), ("base", "-", "base"),
                                   ("plus", "--", "+200 m")):
            s = pd.read_csv(PROCESSED / f"threshold_levels_{year}_{variant}.csv")
            s = s[np.isfinite(s["threshold"])]
            ax.plot(s["level"], s["median_norm_gjsd"], ls=ls, lw=1.2,
                    color=viz.YEAR_COLORS[year], label=label,
                    solid_capstyle="round")
        ax.set_xticks(range(1, len(PROFILE_LEVELS) + 1))
        ax.set_xticklabels([LEVEL_LABELS[t] for t in PROFILE_LEVELS],
                           fontsize=6)
        ax.set_xlabel("hierarchy level (base threshold)")
        ax.set_ylabel("median normalised GJSD")
        viz.panel_label(ax, "c" if year == "2010" else "d",
                        f"threshold perturbation ({year})")
        ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    viz.save(fig, FIGS, "robustness")


def fig_methodology():
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_hex, to_rgb
    from matplotlib.patches import FancyBboxPatch
    from shapely import from_wkt

    def tint(color, f):
        """Blend toward white (f > 0) or black (f < 0)."""
        rgb = np.asarray(to_rgb(color))
        rgb = rgb + (1 - rgb) * f if f > 0 else rgb * (1 + f)
        return to_hex(rgb)

    mem = pd.read_parquet(INTERIM / "percolation" / "memberships.parquet")
    coords = nodes_lonlat()
    x0, x1, y0, y1 = -79.12, -78.92, -1.72, -1.50
    box = coords[(coords["lon"].between(x0, x1)) & (coords["lat"].between(y0, y1))]

    T1, T2, T3 = 160.0, 1080.0, 4070.0
    lab = {t: mem[mem["threshold"] == t].set_index("node_id")["cluster"]
              .reindex(box.index).dropna() for t in (T1, T2, T3)}
    NEUTRAL = "#4a4944"
    REGION = viz.CAT[5]

    parents = []
    for slot, (pid, n) in enumerate(lab[T2].value_counts().head(3).items()):
        hue = viz.CAT[slot]
        kids = lab[T1].reindex(lab[T2].index[lab[T2] == pid]).value_counts()
        kids = kids[kids >= 40].head(2)
        if len(kids) >= 2:
            shades = [tint(hue, -0.30), tint(hue, 0.45)]
        else:
            shades = [hue]
        parents.append({"id": pid, "n": n, "hue": hue,
                        "kids": list(zip(kids.index, kids, shades))})
    root_id, root_n = next(iter(lab[T3].value_counts().items()))

    fig = plt.figure(figsize=(viz.DOUBLE_COL, 3.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=0.16, wspace=0.1)
    maps = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:])
    for ax in (*maps, ax_c, ax_d):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    STRIP = 0.03
    for ax in maps:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0 - STRIP, y1)
        ax.set_aspect("equal")

    ax = maps[0]
    e = pd.read_csv(ROOT / CFG["network"]["edgelist"], header=None,
                    names=["source", "target", "length_m"])
    # commas inside the WKT LINESTRINGs rule out a plain csv parse
    raw = (ROOT / "data/network/ec_edges.csv").read_text().splitlines()
    e["wkt"] = pd.Series(raw).str.split(", ", n=2).str[2].str.strip()
    e = e[e["source"].isin(box.index) & e["target"].isin(box.index)]

    def geom_segs(edges):
        return [np.asarray(g.coords) for g in from_wkt(edges["wkt"].to_numpy())]

    ax.add_collection(LineCollection(geom_segs(e), colors=viz.INK_2,
                                     linewidths=0.3, rasterized=True))
    ax.set_title("street network", fontsize=7, color=viz.INK, pad=3)
    viz.panel_label(ax, "a")
    bar_deg = 5 / (111.32 * np.cos(np.radians(-1.6)))
    bx, by = x0 + 0.004, y0 - STRIP / 2
    ax.plot([bx, bx + bar_deg], [by, by], color=viz.INK_2, lw=1.0,
            solid_capstyle="butt")
    ax.annotate("5 km", (bx + bar_deg + 0.008, by), ha="left",
                va="center", fontsize=5.5, color=viz.INK_2)

    colours = [
        {kid: shade for p in parents for kid, _, shade in p["kids"]},
        {p["id"]: p["hue"] for p in parents},
        {root_id: REGION},
    ]
    for ax, t, cmap_t in zip(maps[1:], (T1, T2, T3), colours):
        ax.scatter(box["lon"], box["lat"], s=0.4, color=viz.GRID,
                   linewidths=0, rasterized=True)
        for cl, colour in cmap_t.items():
            ids = lab[t].index[lab[t] == cl]
            ee = e[(e["length_m"] <= t) & e["source"].isin(ids)
                   & e["target"].isin(ids)]
            if not ee.empty:
                ax.add_collection(LineCollection(geom_segs(ee), colors=colour,
                                                 linewidths=0.35,
                                                 rasterized=True, zorder=2))
            pts = box.reindex(ids)
            ax.scatter(pts["lon"], pts["lat"], s=1.2, color=colour,
                       linewidths=0, rasterized=True, zorder=3)
        ax.set_title(f"δ = {LEVEL_LABELS[int(t)]}", fontsize=7, color=viz.INK,
                     pad=3)
        ax.annotate("", xy=(-0.02, 0.5), xytext=(-0.20, 0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=viz.INK_2, lw=1.0,
                                    mutation_scale=8, shrinkA=0, shrinkB=0))
    viz.panel_label(maps[1], "b")

    ax = ax_c
    Y1, Y2, Y3, YC = 0.0, 1.05, 2.1, 2.85

    def node(x, y, n, colour):
        ax.scatter([x], [y], s=2.1 * np.sqrt(n), color=colour, zorder=3,
                   edgecolor=viz.SURFACE, linewidth=0.4)

    def edge(xa, ya, xb, yb, **kw):
        ax.plot([xa, xb], [ya, yb], color=viz.BASELINE, lw=0.8, zorder=1, **kw)

    x_leaf, leaf_step, group_gap = 0.06, 0.16, 0.04
    first_split = None
    parent_xs = []
    for p in parents:
        kid_xs = []
        for kid, n, shade in p["kids"]:
            kid_xs.append((x_leaf, n, shade))
            x_leaf += leaf_step
        px = float(np.mean([k[0] for k in kid_xs]))
        parent_xs.append(px)
        for kx, kn, kshade in kid_xs:
            edge(px, Y2, kx, Y1)
            node(kx, Y1, kn, kshade)
        node(px, Y2, p["n"], p["hue"])
        if first_split is None:
            first_split = (px, kid_xs)
        x_leaf += group_gap
    x_right = x_leaf - leaf_step - group_gap
    xr = float(np.mean(parent_xs))
    for px in parent_xs:
        edge(xr, Y3, px, Y2)
    node(xr, Y3, root_n, REGION)
    edge(xr, Y3 + 0.18, xr, YC - 0.22, ls=(0, (2, 2)))
    node(xr, YC, 1.3 * root_n, NEUTRAL)
    for yy, t in ((Y1, T1), (Y2, T2), (Y3, T3)):
        ax.annotate(f"δ = {LEVEL_LABELS[int(t)]}", (-0.06, yy), ha="right",
                    va="center", fontsize=6, color=viz.INK_2)
    x_lab = x_right + 0.10
    for yy, semantic in ((Y1, "cities"), (Y2, "conurbations"), (Y3, "region"),
                         (YC, "country (root)")):
        ax.annotate(semantic, (x_lab, yy), ha="left", va="center", fontsize=6,
                    color=viz.MUTED)
    hx0 = first_split[1][0][0] - 0.075
    hx1 = first_split[1][-1][0] + 0.075
    ax.add_patch(FancyBboxPatch((hx0, Y1 - 0.34), hx1 - hx0, Y2 - Y1 + 0.70,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                ls=(0, (3, 2)), lw=0.7, edgecolor=viz.MUTED,
                                facecolor="none", zorder=2))
    ax.annotate("split shown in d", ((hx0 + hx1) / 2, Y1 - 0.62), ha="center",
                va="top", fontsize=5.5, color=viz.MUTED)
    ax.set_xlim(-0.34, x_lab + 0.36)
    ax.set_ylim(-1.05, 3.3)
    viz.panel_label(ax, "c", "hierarchical tree")

    ax = ax_d
    (k1x, k1n, k1shade), (k2x, k2n, k2shade) = first_split[1]
    pi1 = k1n / (k1n + k2n)
    w = np.linspace(0.3, 1.8, 10)
    c1 = w / w.sum()
    c2 = w[::-1] / w.sum()
    M = pi1 * c1 + (1 - pi1) * c2
    YM, YCH, H = 2.05, 1.2, 0.42
    GAP = 0.008

    def comp_bar(yc, comp, left, width):
        for g, share in enumerate(comp):
            ax.barh(yc, share * width, left=left, height=H,
                    color=viz.SEQ(0.15 + 0.7 * g / 9), linewidth=0.4,
                    edgecolor=viz.SURFACE)
            left += share * width

    comp_bar(YM, M, 0.0, 1.0)
    comp_bar(YCH, c1, 0.0, pi1 - GAP)
    comp_bar(YCH, c2, pi1 + GAP, 1 - pi1 - GAP)
    for gx in (0.0, pi1, 1.0):
        ax.plot([gx, gx], [YM - H / 2, YCH + H / 2], color=viz.GRID, lw=0.6,
                ls=(0, (2, 2)), zorder=0)
    hue = parents[0]["hue"]
    ax.scatter([-0.10], [YM], s=16, color=hue, edgecolor=viz.SURFACE,
               linewidth=0.4)
    ax.annotate("$M$", (-0.06, YM), ha="left", va="center", fontsize=7,
                color=viz.INK_2)
    YL = YCH - 0.52
    for xc, shade, plab in (((pi1 - GAP) / 2, k1shade, "$\\vec{P}_1,\\ \\pi_1$"),
                            ((1 + pi1 + GAP) / 2, k2shade,
                             "$\\vec{P}_2,\\ \\pi_2$")):
        ax.scatter([xc - 0.10], [YL], s=13, color=shade, edgecolor=viz.SURFACE,
                   linewidth=0.4)
        ax.annotate(plab, (xc - 0.06, YL), ha="left", va="center", fontsize=6.5,
                    color=viz.INK_2)
    ax.annotate("group $b$: low → high", (0.5, YM + H / 2 + 0.1), ha="center",
                va="bottom", fontsize=5.5, color=viz.MUTED)
    ax.annotate("$H(M)$", (1.04, YM), ha="left", va="center", fontsize=6,
                color=viz.MUTED)
    ax.annotate("$\\pi_c H(\\vec{P}_c)$", (1.04, YCH), ha="left", va="center",
                fontsize=6, color=viz.MUTED)
    ax.annotate("$\\mathrm{JSD}_{\\pi} = H(M) - \\sum_c \\pi_c H(\\vec{P}_c)$",
                (0.5, 0.1), ha="center", va="center", fontsize=7,
                color=viz.INK)
    ax.set_xlim(-0.26, 1.36)
    ax.set_ylim(-0.15, 2.85)
    viz.panel_label(ax, "d", "information gain (GJSD)")
    viz.save(fig, FIGS, "methodology")


def fig_exposure():
    """SI: exposure of the lowest/highest ICV quartile at parroquia level, 2010."""
    dw = pd.read_parquet(INTERIM / "units_2010" / "dwellings.parquet",
                         columns=["ICV"]).dropna()
    ids = dw.index.astype(np.int64).astype(str).str.zfill(18)
    parr = ids.str[:6]
    q = pd.qcut(dw["ICV"], 4, labels=False)

    import pyogrio
    gpkg = ("data/raw/geodatabase/GEODATABASE_NACIONAL_2021/"
            "GEODATABASE_NACIONAL_2021.gpkg")
    sectors = pyogrio.read_dataframe(ROOT / gpkg, layer="sec_a", columns=["sec"])
    sectors["sec"] = sectors["sec"].astype(str).str.zfill(12)
    cent = sectors.geometry.centroid.to_crs("EPSG:4326")
    pc = (pd.DataFrame({"parr": sectors["sec"].str[:6], "lon": cent.x,
                        "lat": cent.y})
          .groupby("parr").median())

    df = pd.DataFrame({"parr": parr.values, "q": q.values})
    tot = df.groupby("parr").size()
    fig, axes = plt.subplots(1, 2, figsize=(viz.DOUBLE_COL, 3.2))
    for ax, letter, (quart, name) in zip(axes, "ab", [(0, "lowest quartile"),
                                                      (3, "highest quartile")]):
        nq = df[df["q"] == quart].groupby("parr").size().reindex(tot.index, fill_value=0)
        exposure = (nq / nq.sum()) / (tot / tot.sum())
        d = pc.join(np.log2(exposure.replace(0, np.nan)).rename("v"), how="inner")
        draw_country(ax)
        sc = ax.scatter(d["lon"], d["lat"], s=3.5, c=d["v"], cmap=viz.DIVERGING,
                        vmin=-3, vmax=3, linewidths=0, rasterized=True, zorder=2)
        viz.panel_label(ax, letter, name)
        map_frame(ax)
        if letter == "a":
            viz.scale_bar(ax)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.55, pad=0.01,
                        ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    cbar.set_label("exposure (log scale)")
    viz.save(fig, FIGS, "exposure")


FIGURES = {
    "methodology": fig_methodology,
    "percolation_maps": fig_percolation_maps,
    "hierarchical_tree": fig_hierarchical_tree,
    "gjsd_tree_2010": fig_gjsd_tree_2010,
    "exposure": fig_exposure,
    "icv_panel": fig_icv_panel,
    "region_map": fig_region_map,
    "admin_map": fig_admin_map,
    "robustness": fig_robustness,
    "percolation_curve": fig_percolation_curve,
    "level_profile_delta": fig_level_profile_delta,
    "upsilon_map": fig_upsilon_map,
    "gjsd_profiles": fig_gjsd_profiles,
    "city_profiles": fig_city_profiles,
    "change_map": fig_change_map,
    "level_change_map": fig_level_change_map,
    "tree_gjsd": fig_tree_gjsd,
}


def main(*names: str) -> None:
    viz.use_style()
    for name in names or FIGURES:
        print(f"figure: {name}", flush=True)
        FIGURES[name]()


if __name__ == "__main__":
    main(*sys.argv[1:])
