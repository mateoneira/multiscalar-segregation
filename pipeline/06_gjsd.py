"""Step 06: GJSD per tree node, level summary, and characteristic scales.

Usage: 06_gjsd.py <units-dirname> <output-suffix> [<labels-file>]
e.g.   06_gjsd.py units_2010 2010 unit_labels_sec2022.parquet
       06_gjsd.py units_2022 2022 unit_labels_sec2022.parquet
"""

import sys

import networkx as nx
import pandas as pd

from multiscalar_segregation import analysis, load_config, repo_root, tree


def main(units_dir: str = "units_2010", suffix: str = "2010",
         labels_file: str = "unit_labels_sec2022.parquet") -> None:
    cfg = load_config()
    root = repo_root()
    interim = root / cfg["paths"]["data_interim"]
    processed = root / cfg["paths"]["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    n_groups = int(cfg["groups"]["n_groups"])

    G = nx.read_graphml(interim / "tree" / "tree.graphml")
    unit_table = pd.read_parquet(interim / units_dir / "unit_counts.parquet")
    unit_counts = unit_table[[f"g{i}" for i in range(n_groups)]].astype(float)

    labels_long = pd.read_parquet(interim / "tree" / labels_file)
    unit_labels = {
        int(level): g.set_index("unit_id")["label"]
        for level, g in labels_long.groupby("level")
    }
    root_level = max(unit_labels) + 1
    unit_labels[root_level] = pd.Series(tree.ROOT, index=unit_counts.index)

    table = analysis.gjsd_table(G, unit_labels, unit_counts)
    table.to_csv(processed / f"gjsd_nodes_{suffix}.csv")

    summary = analysis.level_summary(table)
    summary.to_csv(processed / f"gjsd_levels_{suffix}.csv")
    print(summary.to_string(), flush=True)

    ups = analysis.upsilon(G, table)
    ups.to_csv(processed / f"upsilon_{suffix}.csv")

    best = table["norm_gjsd"].idxmax()
    print(f"\nglobal max norm-GJSD: {best} "
          f"(level {table.loc[best, 'level']}, d={table.loc[best, 'threshold']:g} m, "
          f"norm GJSD={table.loc[best, 'norm_gjsd']:.4f}, "
          f"dwellings={table.loc[best, 'dwellings']:,.0f})")
    print(f"peak median level:\n{summary.loc[summary['median_norm_gjsd'].idxmax()]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
