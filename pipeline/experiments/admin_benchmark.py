"""Experiment: GJSD decomposition on the administrative hierarchy (benchmark).

The paper's central novelty claim is that deriving the hierarchy of scales
from street-network connectivity reveals segregation patterns that a
decomposition over nested *administrative* units would miss. This experiment
tests that claim head-on: it runs the exact same normalised-GJSD machinery on
Ecuador's DPA nesting

    census sector (unit) -> parroquia -> canton -> province -> macro-region
    -> national root

for both census years, and compares (i) the level profiles, (ii) the top
nodes, and (iii) the scale-dependence of the 2010-2022 change against the
percolation-hierarchy results.

Admin codes come from the 2022 sector id (12 digits): province = [:2],
canton = [:4], parroquia = [:6]. Macro-regions follow the standard INEC
grouping (Costa / Sierra / Amazonía / Galápagos). Unlike the percolation
tree, the administrative partition is exhaustive: every unit nests exactly,
so there are no remainder pseudo-children.

Writes data/processed/experiments/admin_gjsd_{nodes,levels}_{year}.csv and
figures/admin_benchmark.(pdf|png) (a main-text figure). Does NOT modify the
pipeline outputs or config.

Usage: python pipeline/experiments/admin_benchmark.py [--figure-only]
    --figure-only rebuilds the figure from the saved CSVs, skipping the
    decomposition.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiscalar_segregation import geography, load_config, metrics, repo_root, viz

CFG = load_config()
ROOT = repo_root()
INTERIM = ROOT / CFG["paths"]["data_interim"]
PROCESSED = ROOT / CFG["paths"]["data_processed"]
OUT = PROCESSED / "experiments"
FIGS = ROOT / CFG["paths"]["figures"]
YEARS = ("2010", "2022")
N_GROUPS = int(CFG["groups"]["n_groups"])
GCOLS = [f"g{i}" for i in range(N_GROUPS)]

REGION = {**geography.PROVINCE_REGION, "20": "Galápagos", "90": "No delimitado"}
PROVINCE_NAMES = {
    "01": "Azuay", "02": "Bolivar", "03": "Canar", "04": "Carchi",
    "05": "Cotopaxi", "06": "Chimborazo", "07": "El Oro", "08": "Esmeraldas",
    "09": "Guayas", "10": "Imbabura", "11": "Loja", "12": "Los Rios",
    "13": "Manabi", "14": "Morona Santiago", "15": "Napo", "16": "Pastaza",
    "17": "Pichincha", "18": "Tungurahua", "19": "Zamora Chinchipe",
    "20": "Galapagos", "21": "Sucumbios", "22": "Orellana",
    "23": "Santo Domingo", "24": "Santa Elena",
}
LEVELS = ["parroquia", "canton", "province", "region", "national"]

GPKG = ROOT / ("data/raw/geodatabase/GEODATABASE_NACIONAL_2021/"
               "GEODATABASE_NACIONAL_2021.gpkg")


def admin_names() -> dict:
    """Human-readable names for parroquia and canton codes (INEC geodatabase)."""
    try:
        import pyogrio
        am = pyogrio.read_dataframe(GPKG, layer="aream_a", read_geometry=False)
    except Exception as e:
        print(f"  (no admin names: {e})")
        return {}
    am["parroquia"] = am["parroquia"].astype(str).str.zfill(6)
    am["canton"] = am["parroquia"].str[:4]
    prio = {"CAPITAL PROVINCIAL": 0, "CABECERA CANTONAL": 0,
            "CABERECA CANTONAL": 0, "CABECERA PARROQUIAL": 1}
    am["prio"] = am["tipo_aream"].map(prio).fillna(2)
    am = am.sort_values("prio")
    names = {}
    names.update(am.drop_duplicates("parroquia")
                 .set_index("parroquia")["nom_aman"].str.title())
    names.update(am[am["prio"] == 0].drop_duplicates("canton")
                 .set_index("canton")["nom_aman"].str.title())
    names.update({p: n for p, n in PROVINCE_NAMES.items()})
    return names


def split_gjsd(child_counts: pd.DataFrame, parent_of_child: pd.Series,
               level: str) -> pd.DataFrame:
    """Normalised GJSD of every parent when split into the given children.

    child_counts : one row per child, columns GCOLS (dwelling counts / group).
    parent_of_child : child id -> parent id.
    Parents with a single non-empty child carry no GJSD (NaN), matching
    analysis.gjsd_table's treatment of trivial splits.
    """
    records = []
    for parent, block in child_counts.groupby(parent_of_child):
        counts = block.to_numpy(dtype=float)
        counts = counts[counts.sum(axis=1) > 0]
        total = counts.sum()
        if total == 0:
            continue
        if len(counts) > 1:
            weights = counts.sum(axis=1) / total
            dists = counts / counts.sum(axis=1, keepdims=True)
            raw = metrics.gjsd(dists, weights)
            upper = metrics.gjsd_max(weights)
            norm = raw / upper if upper > 0 else 0.0
        else:
            raw = upper = norm = np.nan
        records.append({"node": parent, "level": level,
                        "n_children": int(len(counts)),
                        "dwellings": float(total),
                        "gjsd": raw, "gjsd_max": upper, "norm_gjsd": norm})
    return pd.DataFrame.from_records(records).set_index("node")


def admin_tables(year: str) -> pd.DataFrame:
    """GJSD node table over the full administrative hierarchy for one census."""
    units = pd.read_parquet(INTERIM / f"units_{year}" / "unit_counts.parquet")
    sector = units[GCOLS].astype(float)
    sector.index = sector.index.astype(str).str.zfill(12)

    parr = sector.groupby(sector.index.str[:6]).sum()
    cant = parr.groupby(parr.index.str[:4]).sum()
    prov = cant.groupby(cant.index.str[:2]).sum()
    reg = prov.groupby(prov.index.map(REGION)).sum()

    parts = [
        split_gjsd(sector, pd.Series(sector.index.str[:6], index=sector.index),
                   "parroquia"),
        split_gjsd(parr, pd.Series(parr.index.str[:4], index=parr.index),
                   "canton"),
        split_gjsd(cant, pd.Series(cant.index.str[:2], index=cant.index),
                   "province"),
        split_gjsd(prov, pd.Series(prov.index.map(REGION), index=prov.index),
                   "region"),
        split_gjsd(reg, pd.Series("national", index=reg.index), "national"),
    ]
    table = pd.concat(parts)
    table["level"] = pd.Categorical(table["level"], categories=LEVELS,
                                    ordered=True)
    return table


def level_summary(table: pd.DataFrame) -> pd.DataFrame:
    valid = table.dropna(subset=["norm_gjsd"])
    return valid.groupby("level", observed=True).agg(
        n_parents=("norm_gjsd", "size"),
        median_norm_gjsd=("norm_gjsd", "median"),
        mean_norm_gjsd=("norm_gjsd", "mean"),
        median_dwellings=("dwellings", "median"),
    )


def within_node_delta(tables: dict) -> pd.DataFrame:
    """Median within-node 2010->2022 change per admin level (the same series
    the manuscript quotes for the percolation hierarchy)."""
    a, b = tables["2010"], tables["2022"]
    common = a.dropna(subset=["norm_gjsd"]).index.intersection(
        b.dropna(subset=["norm_gjsd"]).index)
    delta = (b.loc[common, "norm_gjsd"] - a.loc[common, "norm_gjsd"])
    lev = a.loc[common, "level"]
    out = pd.DataFrame({
        "n": delta.groupby(lev, observed=True).size(),
        "median_2010": a.dropna(subset=["norm_gjsd"])
                        .groupby("level", observed=True)["norm_gjsd"].median(),
        "median_2022": b.dropna(subset=["norm_gjsd"])
                        .groupby("level", observed=True)["norm_gjsd"].median(),
        "median_delta": delta.groupby(lev, observed=True).median(),
    })
    out["pct_of_2010_median"] = 100 * out["median_delta"] / out["median_2010"]
    return out


def profile_panel(ax, x, med10, med22, deltas, ticklabels):
    for xi, lo, hi in zip(x, med10, med22):
        ax.plot([xi, xi], [lo, hi], color=viz.BASELINE, lw=1.0, zorder=1)
    ax.scatter(x, med10, s=22, color=viz.YEAR_COLORS["2010"], zorder=3,
               label="2010", edgecolor=viz.SURFACE, linewidth=0.6)
    ax.scatter(x, med22, s=22, color=viz.YEAR_COLORS["2022"], zorder=3,
               label="2022", edgecolor=viz.SURFACE, linewidth=0.6)
    for xi, lo, hi, pct in zip(x, med10, med22, deltas):
        ax.annotate(f"{pct:.0f}%".replace("-", "\N{MINUS SIGN}"),
                    xy=(xi, min(lo, hi)), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=6,
                    color=viz.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("median normalised GJSD")


def make_figure(tables, summaries, admin_delta):
    perc_delta = pd.read_csv(PROCESSED / "gjsd_levels_delta.csv", index_col=0)
    perc_labels = {**{t: viz.level_label(t) for t in CFG["percolation"]["levels_m"]},
                   np.inf: "national"}

    fig, axes = plt.subplots(1, 3, figsize=(viz.DOUBLE_COL, 2.5))

    p = perc_delta
    profile_panel(axes[0], np.arange(len(p)), p["median_2010"],
                  p["median_2022"],
                  100 * p["median_delta"] / p["median_2010"],
                  [perc_labels[t] for t in p["threshold"]])
    viz.panel_label(axes[0], "a", "percolation hierarchy")
    axes[0].set_xlabel("hierarchy level (percolation threshold)")
    axes[0].legend(loc="upper left", handletextpad=0.4)

    d = admin_delta
    profile_panel(axes[1], np.arange(len(d)), d["median_2010"],
                  d["median_2022"], d["pct_of_2010_median"],
                  [{"parroquia": "parroquia", "canton": "cantón",
                    "province": "province", "region": "region",
                    "national": "national"}[l] for l in d.index])
    viz.panel_label(axes[1], "b", "administrative hierarchy")
    axes[1].set_xlabel("administrative level")
    axes[1].set_ylabel("")

    ymax = 1.05 * max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]:
        ax.set_ylim(0, ymax)

    bins = np.geomspace(20, 6e6, 12)
    colors = {"percolation": viz.CAT[0], "administrative": viz.CAT[5]}
    for year, ls in (("2010", "-"), ("2022", "--")):
        perc_nodes = pd.read_csv(PROCESSED / f"gjsd_nodes_{year}.csv",
                                 index_col=0).dropna(subset=["norm_gjsd"])
        nodes = {"percolation": perc_nodes,
                 "administrative": tables[year].dropna(subset=["norm_gjsd"])}
        for which, d in nodes.items():
            if year == "2010":
                axes[2].scatter(d["dwellings"], d["norm_gjsd"], s=2.5,
                                color=colors[which], alpha=0.16, linewidths=0,
                                rasterized=True, zorder=1)
            grp = pd.cut(np.log(d["dwellings"]), np.log(bins))
            med = d.groupby(grp, observed=True).agg(
                x=("dwellings", "median"), y=("norm_gjsd", "median"),
                n=("norm_gjsd", "size"))
            med = med[med["n"] >= 5]
            axes[2].plot(med["x"], med["y"], ls=ls, color=colors[which],
                         lw=1.2, marker="o", ms=2.5, zorder=3,
                         label=f"{which} {year}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("parent node size (dwellings)")
    axes[2].set_ylim(0, 0.3)
    viz.panel_label(axes[2], "c", "same-size comparison")
    handles, labels = axes[2].get_legend_handles_labels()
    order = [0, 2, 1, 3]
    axes[2].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper right", fontsize=5.8, handlelength=3.2,
                   labelspacing=0.3)

    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    viz.save(fig, FIGS, "admin_benchmark")


def main() -> None:
    viz.use_style()
    OUT.mkdir(parents=True, exist_ok=True)

    if "--figure-only" in sys.argv[1:]:
        tables = {y: pd.read_csv(OUT / f"admin_gjsd_nodes_{y}.csv",
                                 index_col=0) for y in YEARS}
        delta = pd.read_csv(OUT / "admin_gjsd_levels_delta.csv", index_col=0)
        make_figure(tables, None, delta)
        return

    names = admin_names()

    tables, summaries = {}, {}
    for year in YEARS:
        t = admin_tables(year)
        t["name"] = [names.get(n, n) for n in t.index]
        t.to_csv(OUT / f"admin_gjsd_nodes_{year}.csv")
        s = level_summary(t)
        s.to_csv(OUT / f"admin_gjsd_levels_{year}.csv")
        tables[year], summaries[year] = t, s
        print(f"\n=== {year}: administrative-hierarchy level summary ===")
        print(s.to_string())

        top = (t.dropna(subset=["norm_gjsd"])
               .sort_values("norm_gjsd", ascending=False).head(12))
        print(f"\n{year}: top 12 admin nodes by normalised GJSD")
        print(top[["level", "name", "n_children", "dwellings",
                   "norm_gjsd"]].to_string())

        valid = t.dropna(subset=["norm_gjsd"])
        hits = valid[valid["name"].str.contains("Guaranda|Guanujo", case=False,
                                                na=False)]
        for node, row in hits.iterrows():
            rank = int((valid["norm_gjsd"] > row["norm_gjsd"]).sum()) + 1
            print(f"  {row['level']} {node} ({row['name']}): norm GJSD "
                  f"{row['norm_gjsd']:.3f} (rank {rank}/{len(valid)})")

    delta = within_node_delta(tables)
    delta.to_csv(OUT / "admin_gjsd_levels_delta.csv")
    print("\n=== 2010->2022 within-node change per admin level ===")
    print(delta.to_string())

    perc = pd.read_csv(PROCESSED / "gjsd_levels_delta.csv", index_col=0)
    print("\n(for reference, percolation hierarchy)")
    print(perc.assign(pct=100 * perc["median_delta"] / perc["median_2010"])
          .to_string())

    make_figure(tables, summaries, delta)


if __name__ == "__main__":
    main()
