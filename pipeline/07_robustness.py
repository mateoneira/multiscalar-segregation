"""Step 07: robustness analyses (SI).

Usage: 07_robustness.py <units-dirname> <suffix>
e.g.   07_robustness.py units_2010 2010

Runs: (a) 1,000-permutation null model for node-level norm-GJSD;
(b) group-count sensitivity (quartiles/quintiles/deciles) with Spearman rank
correlations and Upsilon stability; (c) threshold perturbation (±200 m) with
the peak-segregation level under each variant.
"""

import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multiscalar_segregation import (analysis, assignment, geography, load_config,
                                     metrics, percolation, repo_root, robustness,
                                     tree)


def load_tree_and_labels(interim, labels_file):
    G = nx.read_graphml(interim / "tree" / "tree.graphml")
    labels_long = pd.read_parquet(interim / "tree" / labels_file)
    unit_labels = {int(lv): g.set_index("unit_id")["label"]
                   for lv, g in labels_long.groupby("level")}
    return G, unit_labels


def with_root(unit_labels, unit_index):
    out = dict(unit_labels)
    out[max(out) + 1] = pd.Series(tree.ROOT, index=unit_index)
    return out


def null_model(cfg, G, unit_labels, unit_counts, processed, suffix):
    n_perm = int(cfg["robustness"]["n_permutations"])
    seed = int(cfg["robustness"]["random_seed"])
    counts = unit_counts.to_numpy()
    engine = robustness.GjsdEngine(G, unit_labels, unit_counts.index)
    observed = engine.norm_gjsd(counts)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, len(observed)))
    for i in range(n_perm):
        perm = robustness.permuted_unit_counts(unit_counts, rng)
        draws[i] = engine.norm_gjsd(perm.to_numpy()).to_numpy()
        if (i + 1) % 100 == 0:
            print(f"  permutation {i + 1}/{n_perm}", flush=True)

    with np.errstate(invalid="ignore"):
        exceed = np.nansum(draws >= observed.to_numpy(), axis=0)
    n_valid = np.isfinite(draws).sum(axis=0)
    out = pd.DataFrame({
        "observed": observed,
        "null_mean": np.nanmean(draws, axis=0),
        "null_lo": np.nanpercentile(draws, 2.5, axis=0),
        "null_hi": np.nanpercentile(draws, 97.5, axis=0),
        "p_value": (exceed + 1) / (n_valid + 1),
    })
    out.to_csv(processed / f"null_model_{suffix}.csv")
    sig = (out["p_value"] < 0.05).mean()
    print(f"null model: {sig:.1%} of nodes significant at p<0.05; "
          f"median null {out['null_mean'].median():.4f} vs observed "
          f"{out['observed'].median():.4f}")


def group_sensitivity(cfg, G, unit_labels_noroot, units_dir, processed, suffix):
    dwellings = pd.read_parquet(units_dir / "dwellings.parquet",
                                columns=["ICV", "num_people"])
    dwellings = dwellings.dropna(subset=["ICV"])
    width = geography.ID_VIV_DIGITS[suffix]
    sector = (dwellings.index // 10 ** (width - 12)).astype(np.int64) \
        .astype(str).str.zfill(12)

    results = {}
    for n in cfg["groups"]["sensitivity"]:
        counts = robustness.regroup_unit_counts(dwellings["ICV"],
                                                pd.Series(sector.values), int(n))
        labels = with_root(unit_labels_noroot, counts.index)
        table = analysis.gjsd_table(G, labels, counts.astype(float))
        ups = analysis.upsilon(G, table)
        results[int(n)] = (table["norm_gjsd"], ups["upsilon_threshold"])
        table.to_csv(processed / f"gjsd_nodes_{suffix}_g{n}.csv")

    rows = []
    ns = sorted(results)
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            ga, ua = results[a]
            gb, ub = results[b]
            j = pd.concat([ga, gb], axis=1, keys=["a", "b"]).dropna()
            rho = spearmanr(j["a"], j["b"]).statistic
            u = pd.concat([ua, ub], axis=1, keys=["a", "b"]).dropna()
            same = (u["a"] == u["b"]).mean()
            rows.append({"groups_a": a, "groups_b": b, "spearman_gjsd": rho,
                         "upsilon_same_level": same})
            print(f"  {a} vs {b} groups: Spearman GJSD {rho:.4f}, "
                  f"same Upsilon level {same:.1%}")
    pd.DataFrame(rows).to_csv(processed / f"group_sensitivity_{suffix}.csv", index=False)


def threshold_sensitivity(cfg, unit_counts, node_to_unit, processed, suffix):
    root = repo_root()
    base_levels = [float(t) for t in cfg["percolation"]["levels_m"]]
    delta = float(cfg["robustness"]["threshold_perturbation_m"])
    sweep_min = float(cfg["percolation"]["sweep_min_m"])
    variants = {"base": base_levels,
                "minus": [max(sweep_min, t - delta) for t in base_levels],
                "plus": [t + delta for t in base_levels]}

    nodes = pd.read_csv(root / cfg["network"]["nodes"], header=None,
                        names=["node_id", "lon", "lat"])
    edges, eu = percolation.load_edgelist(str(root / cfg["network"]["edgelist"]))
    universe = np.union1d(nodes["node_id"].values, eu)
    all_thresholds = sorted({t for v in variants.values() for t in v})
    res = percolation.percolate(edges, universe, snapshot_thresholds=all_thresholds,
                                curve_step=10, curve_max=max(all_thresholds))

    rows = []
    for name, levels in variants.items():
        memberships = {t: res.membership_series(t) for t in levels}
        G = tree.build_tree(memberships,
                            min_cluster_nodes=int(cfg["percolation"]["min_cluster_nodes"]))
        labels_noroot = {}
        long = assignment.compute_unit_labels(G, memberships, node_to_unit)
        for lv, g in long.groupby("level"):
            labels_noroot[int(lv)] = g.set_index("unit_id")["label"]
        labels = with_root(labels_noroot, unit_counts.index)
        table = analysis.gjsd_table(G, labels, unit_counts)
        summary = analysis.level_summary(table)
        summary.to_csv(processed / f"threshold_levels_{suffix}_{name}.csv")
        peak = summary["median_norm_gjsd"].idxmax()
        rows.append({"variant": name, "levels": levels,
                     "peak_level": int(peak),
                     "peak_threshold": float(summary.loc[peak, "threshold"]),
                     "peak_median_gjsd": float(summary.loc[peak, "median_norm_gjsd"])})
        print(f"  {name}: peak at level {peak} "
              f"(d={summary.loc[peak, 'threshold']:g} m)")
    pd.DataFrame(rows).to_csv(processed / f"threshold_sensitivity_{suffix}.csv",
                              index=False)


def main(units_dirname: str, suffix: str, only: str = "all") -> None:
    cfg = load_config()
    root = repo_root()
    interim = root / cfg["paths"]["data_interim"]
    processed = root / cfg["paths"]["data_processed"]
    units_dir = interim / units_dirname

    G, unit_labels_noroot = load_tree_and_labels(interim, "unit_labels_sec2022.parquet")
    unit_table = pd.read_parquet(units_dir / "unit_counts.parquet")
    n_groups = int(cfg["groups"]["n_groups"])
    unit_counts = unit_table[[f"g{i}" for i in range(n_groups)]].astype(float)
    unit_labels = with_root(unit_labels_noroot, unit_counts.index)

    if only in ("all", "null"):
        print("=== null model", flush=True)
        null_model(cfg, G, unit_labels, unit_counts, processed, suffix)
    if only in ("all", "groups"):
        print("=== group-count sensitivity", flush=True)
        group_sensitivity(cfg, G, unit_labels_noroot, units_dir, processed, suffix)
    if only not in ("all", "thresholds"):
        return
    print("=== threshold sensitivity", flush=True)
    node_to_unit = pd.read_parquet(interim / "tree" / "node_to_unit_sec2022.parquet")
    threshold_sensitivity(cfg, unit_counts, node_to_unit, processed, suffix)


if __name__ == "__main__":
    main(*sys.argv[1:])
