"""Experiment: quantify the connectivity-segregation link.

The abstract claims that "regions that connect late in the percolation
hierarchy are also among the most segregated", but the manuscript supports
this only with the Amazon anecdote in the Discussion. This experiment makes
the claim quantitative.

For every city (leaf of the connectivity hierarchy) we define its
*connection scale*: the threshold of the first strict ancestor with at least
two retained children, i.e. the smallest retained threshold at which the
city's street-network component participates in a macroscopic merge. Cities
embedded in dense, well-connected fabric merge at a few hundred metres;
infrastructurally isolated towns stay alone until the regional or national
assembly of the network (the national root closes at 6,357 m).

Against the connection scale we test, per census year:
  - the city's peak normalised GJSD along its path to the root (its
    segregation at its characteristic scale Upsilon);
  - the normalised GJSD of the connection node itself (how divergent the
    city is from what it finally connects to);
  - the city's mean ICV (is infrastructural isolation also deprivation?).
Spearman rank correlations; a node-level correlation (threshold vs norm-GJSD
across genuine-merge parent nodes) is reported alongside, since leaves that
share an ancestor share its GJSD value (pseudo-replication).

Writes data/processed/experiments/connectivity_segregation_{leaves,stats,tails}.csv
and figures/connectivity_segregation.(pdf|png) (an SI figure). Does NOT modify
the pipeline outputs or config.

Usage: python pipeline/experiments/connectivity_segregation.py
"""

import importlib.util

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multiscalar_segregation import geography, load_config, repo_root, tree, viz

CFG = load_config()
ROOT = repo_root()
INTERIM = ROOT / CFG["paths"]["data_interim"]
PROCESSED = ROOT / CFG["paths"]["data_processed"]
OUT = PROCESSED / "experiments"
FIGS = ROOT / CFG["paths"]["figures"]
YEARS = ("2010", "2022")

NATIONAL_MERGE_M = 6357.0
LEVEL_LABELS = {**{int(t): viz.level_label(t) for t in CFG["percolation"]["levels_m"][1:]},
                NATIONAL_MERGE_M: "national"}

REGION = {**geography.PROVINCE_REGION, "20": "Galápagos", "90": "No delimitado"}
REGION_COLOR = {"Costa": viz.CAT[0], "Sierra": viz.CAT[2],
                "Amazonía": viz.CAT[3], "Galápagos": viz.CAT[6],
                "No delimitado": viz.MUTED}


def load_figs_module():
    """Reuse leaf_geography()/leaf_names() from pipeline/08_figures.py."""
    spec = importlib.util.spec_from_file_location(
        "figs", ROOT / "pipeline" / "08_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def connection_node(G: nx.DiGraph, leaf: str) -> str:
    """First strict ancestor with >= 2 retained children."""
    for anc in tree.ancestry(G, leaf)[1:]:
        if G.out_degree(anc) >= 2:
            return anc
    return tree.ROOT


def leaf_table(G: nx.DiGraph, figs) -> pd.DataFrame:
    """One row per leaf city: connection scale, region, size, ICV, GJSD."""
    _, geo = figs.leaf_geography()
    names = figs.leaf_names(G, geo)
    labels = pd.read_parquet(INTERIM / "tree" / "unit_labels_sec2022.parquet")

    rows = []
    for leaf in tree.leaves(G):
        conn = connection_node(G, leaf)
        t = G.nodes[conn]["threshold"] if conn != tree.ROOT else np.inf
        rows.append({"leaf": leaf, "name": names[leaf],
                     "leaf_level": G.nodes[leaf]["level"],
                     "connection_node": conn,
                     "connection_threshold": min(float(t), NATIONAL_MERGE_M)})
    d = pd.DataFrame(rows).set_index("leaf")
    d = d.join(geo[[c for c in geo.columns if c.startswith("dwellings_")]])

    for year in YEARS:
        units = pd.read_parquet(INTERIM / f"units_{year}" / "unit_counts.parquet")
        table = pd.read_csv(PROCESSED / f"gjsd_nodes_{year}.csv", index_col=0)
        ups = pd.read_csv(PROCESSED / f"upsilon_{year}.csv", index_col=0)
        for leaf in d.index:
            lab = labels[(labels["level"] == d.loc[leaf, "leaf_level"])
                         & (labels["label"] == leaf)]
            u = units.reindex(lab["unit_id"]).dropna(subset=["mean_icv"])
            w = u["num_dwellings"]
            d.loc[leaf, f"mean_icv_{year}"] = float((u["mean_icv"] * w).sum()
                                                    / w.sum())
            if year == YEARS[0]:
                prov = u.index.astype(str).str.zfill(12).str[:2]
                by_reg = w.groupby(prov.map(REGION)).sum()
                d.loc[leaf, "region"] = by_reg.idxmax()
            d.loc[leaf, f"peak_norm_gjsd_{year}"] = \
                ups["upsilon_norm_gjsd"].get(leaf, np.nan)
            d.loc[leaf, f"upsilon_threshold_{year}"] = \
                ups["upsilon_threshold"].get(leaf, np.nan)
            conn = d.loc[leaf, "connection_node"]
            d.loc[leaf, f"conn_norm_gjsd_{year}"] = \
                table["norm_gjsd"].get(conn, np.nan)
    return d


def stats_table(d: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations of everything vs the connection scale."""
    records = []
    for year in YEARS:
        for col, what in [(f"peak_norm_gjsd_{year}", "peak norm GJSD (Upsilon)"),
                          (f"conn_norm_gjsd_{year}", "norm GJSD at connection node"),
                          (f"upsilon_threshold_{year}", "characteristic scale Upsilon (m)"),
                          (f"mean_icv_{year}", "mean ICV")]:
            v = d[["connection_threshold", col]].dropna()
            rho, p = spearmanr(v["connection_threshold"], v[col])
            records.append({"year": year, "against": what, "unit": "city",
                            "n": len(v), "spearman_rho": rho, "p": p})
        table = pd.read_csv(PROCESSED / f"gjsd_nodes_{year}.csv", index_col=0)
        merges = table[(table["n_children"] >= 2)].dropna(subset=["norm_gjsd"])
        rho, p = spearmanr(merges["threshold"], merges["norm_gjsd"])
        records.append({"year": year, "against": "norm GJSD (merge nodes)",
                        "unit": "node", "n": len(merges),
                        "spearman_rho": rho, "p": p})
    return pd.DataFrame.from_records(records)


def tail_enrichment(d: pd.DataFrame) -> pd.DataFrame:
    """The claim is about the top tail ("among the most segregated"), not a
    monotone trend: are late-connecting cities (>= 4.07 km) over-represented
    in the top quartile of peak norm GJSD (and the bottom quartile of ICV)?
    Fisher exact test on the 2x2 table."""
    from scipy.stats import fisher_exact
    late = d["connection_threshold"] >= 4070
    records = []
    for year in YEARS:
        for col, top, what in [
                (f"peak_norm_gjsd_{year}", True, "top-quartile peak GJSD"),
                (f"mean_icv_{year}", False, "bottom-quartile mean ICV")]:
            v = d[col].dropna()
            q = v.quantile(0.75 if top else 0.25)
            tail = (v >= q) if top else (v <= q)
            tab = pd.crosstab(late.reindex(v.index), tail)
            odds, p = fisher_exact(tab.to_numpy())
            records.append({
                "year": year, "tail": what,
                "late_in_tail": int((late.reindex(v.index) & tail).sum()),
                "n_tail": int(tail.sum()),
                "late_share_overall": float(late.mean()),
                "odds_ratio": odds, "fisher_p": p})
    return pd.DataFrame.from_records(records)


def jittered_x(d: pd.DataFrame, rng) -> pd.Series:
    xpos = {t: i for i, t in enumerate(sorted(LEVEL_LABELS))}
    return (d["connection_threshold"].map(xpos)
            + rng.uniform(-0.22, 0.22, size=len(d)))


def scatter_panel(ax, d, x, ycol, year, size, annotate=()):
    for reg, block in d.groupby("region"):
        ax.scatter(x[block.index], block[ycol], s=size[block.index],
                   color=REGION_COLOR[reg], alpha=0.85,
                   edgecolor=viz.SURFACE, linewidth=0.4,
                   label=reg, zorder=3)
    import matplotlib.patheffects as pe
    for leaf in annotate:
        if leaf in d.index and np.isfinite(d.loc[leaf, ycol]):
            left = x[leaf] > 5.2
            ax.annotate(d.loc[leaf, "name"], (x[leaf], d.loc[leaf, ycol]),
                        xytext=(-9 if left else 9, 7),
                        textcoords="offset points",
                        ha="right" if left else "left",
                        fontsize=5.5, color=viz.INK_2, zorder=6,
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=viz.MUTED,
                                        shrinkA=1, shrinkB=3),
                        path_effects=[pe.withStroke(linewidth=1.4,
                                                    foreground=viz.SURFACE)])
    v = d[["connection_threshold", ycol]].dropna()
    rho, p = spearmanr(v["connection_threshold"], v[ycol])
    ax.annotate(f"Spearman ρ = {rho:.2f}, p = {p:.1e}".replace("-", "\N{MINUS SIGN}"),
                xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.5, color=viz.INK)
    ticks = sorted(LEVEL_LABELS)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels([LEVEL_LABELS[t] for t in ticks])
    ax.set_xlim(-0.5, len(ticks) - 0.5)


def make_figure(d: pd.DataFrame) -> None:
    rng = np.random.default_rng(int(CFG["robustness"]["random_seed"]))
    x = jittered_x(d, rng)
    fig, axes = plt.subplots(2, 2, figsize=(viz.DOUBLE_COL, 4.6),
                             sharex=True, sharey="row")
    for j, year in enumerate(YEARS):
        size = 5 + 30 * np.sqrt(d[f"dwellings_{year}"]
                                / d[f"dwellings_{year}"].max())
        ycol = f"peak_norm_gjsd_{year}"
        highest = d[ycol].nlargest(3).index
        annotate = set(["0-124263", "0-41239", "0-286115"]) | set(highest)
        scatter_panel(axes[0, j], d, x, ycol, year, size, annotate)
        viz.panel_label(axes[0, j], "ab"[j], year)

        scatter_panel(axes[1, j], d, x, f"mean_icv_{year}", year, size,
                      annotate=d[f"mean_icv_{year}"].nsmallest(2).index)
        viz.panel_label(axes[1, j], "cd"[j], year)
        axes[1, j].set_xlabel("connection scale (first macroscopic merge)")
    for row in range(2):
        lo, hi = axes[row, 0].get_ylim()
        axes[row, 0].set_ylim(lo, hi + 0.14 * (hi - lo))
    axes[0, 0].set_ylabel("peak normalised GJSD, Υ")
    axes[1, 0].set_ylabel("mean ICV of the city")
    axes[0, 0].legend(loc="lower right", fontsize=6, handletextpad=0.3,
                      borderaxespad=0.2, labelspacing=0.3)
    fig.subplots_adjust(wspace=0.06, hspace=0.22)
    viz.save(fig, FIGS, "connectivity_segregation")


def main() -> None:
    viz.use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = load_figs_module()
    G = nx.read_graphml(INTERIM / "tree" / "tree.graphml")

    d = leaf_table(G, figs)
    d.to_csv(OUT / "connectivity_segregation_leaves.csv")

    print("cities per connection scale:")
    print(d.groupby("connection_threshold").agg(
        n=("name", "size"),
        median_peak_2010=("peak_norm_gjsd_2010", "median"),
        median_peak_2022=("peak_norm_gjsd_2022", "median"),
        median_icv_2010=("mean_icv_2010", "median"),
        median_icv_2022=("mean_icv_2022", "median")).to_string())

    stats = stats_table(d)
    stats.to_csv(OUT / "connectivity_segregation_stats.csv", index=False)
    print("\nSpearman correlations vs connection scale:")
    print(stats.to_string(index=False))

    tails = tail_enrichment(d)
    tails.to_csv(OUT / "connectivity_segregation_tails.csv", index=False)
    print("\nTail enrichment of late-connecting cities (>= 4.07 km):")
    print(tails.to_string(index=False))

    late = d[d["connection_threshold"] >= 4070]
    print(f"\nlate-connecting cities (>= 4.07 km): {len(late)}")
    print(late.sort_values("peak_norm_gjsd_2010", ascending=False)
          [["name", "region", "connection_threshold",
            "peak_norm_gjsd_2010", "peak_norm_gjsd_2022",
            "mean_icv_2010", "mean_icv_2022"]].head(12).to_string())

    make_figure(d)


if __name__ == "__main__":
    main()
