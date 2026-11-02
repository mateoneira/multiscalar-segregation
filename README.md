# The multiscale structure of socio-spatial segregation

Code for *The multiscale structure of socio-spatial segregation* (Neira, Marin & Arcaute).

Most segregation measures work at one fixed scale or follow administrative boundaries, which hides the scales where social divisions operate. This repository implements percolation divergence trees, a framework that measures segregation across the whole hierarchy of spatial scales in a country:

1. **Percolation on the street network.** Sweeping a distance threshold over the national road network merges settlements into larger and larger clusters. The result is a hierarchical tree of the spatial system, running from neighbourhoods to macro-regions.
2. **Information-theoretic decomposition.** At each level of the tree, the Generalised Jensen–Shannon Divergence (GJSD) measures how much information about socio-economic composition we gain by splitting a cluster into its children. The GJSD at a node measures segregation at that node's scale.

We apply the framework to Ecuador using a census-derived index of life conditions (ICV) for the 2010 and 2022 censuses.

## System requirements

- **Operating system:** any platform (Linux, macOS, Windows). Tested on Linux (x86_64).
- **Python:** 3.11–3.14. Tested with Python 3.13.15.
- **Dependencies:** listed in `pyproject.toml`, with exact versions locked in `uv.lock`. The main ones are OSMnx 2.1.0, NetworkX 3.6.1, NumPy 2.4.6, pandas 3.0.3, GeoPandas 1.1.4, Shapely 2.1.2, pyogrio 0.13.0, SciPy 1.17.1 and Matplotlib 3.11.0.
- **Hardware:** the code needs no special hardware, and the demo runs on any laptop. We ran the full analysis on a desktop with 8 CPU cores and 32 GB RAM. It needs about 25 GB of free disk space. The downloads are about 4 GB compressed and 22 GB unpacked.

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and create the environment:

```bash
git clone https://github.com/mateoneira/multiscalar-segregation.git
cd multiscalar-segregation
uv sync                              # creates .venv from uv.lock
uv run pytest                        # 16 tests, about 1 s
```

uv downloads Python 3.13 if it is not already installed. A clean install takes under a minute on a typical broadband connection.

## Demo

`examples/demo.py` runs the full method on a small synthetic urban system and needs no external data. The system is eight grid towns of 100 street intersections each, grouped into pairs and two regions, with 32,000 synthetic dwellings in five socio-economic groups.

```bash
uv run python examples/demo.py
```

Expected output (about 1 s on a normal desktop):

```
15 tree nodes, 800 street nodes, 32000 dwellings

Normalised GJSD per level:
       threshold  n_parents  median_norm_gjsd  mean_norm_gjsd  median_h_index
level
1          500.0          4             0.300           0.309           0.215
2         2500.0          2             0.083           0.083           0.164
3            inf          1             0.292           0.292           0.038

Characteristic scale (Upsilon) per town:
       upsilon_threshold  upsilon_norm_gjsd
leaf
0-0                500.0              0.624
0-100              500.0              0.624
0-200                inf              0.292
...
```

Each row of the first table is a level of the percolation tree; `inf` is the root. Towns 0 and 1 are most segregated from each other at the 500 m level (the pair), while towns 2–5 are most segregated at the scale of the whole system.

## Running on other data

The method needs two inputs:

1. **A street network.** An edge list with `source`, `target` and `length` columns. Any OSMnx graph can be exported to this form.
2. **Population counts.** A table of dwelling (or household) counts per socio-economic group for each spatial unit, plus a mapping from street nodes to units.

`examples/demo.py` shows the core calls in order: `percolation.percolate`, `tree.build_tree`, `analysis.gjsd_table` and `analysis.upsilon`. For census units with polygons, `assignment.compute_unit_labels` assigns each unit to a tree cluster by majority vote of its street nodes. `config.yaml` sets the percolation thresholds, the cluster-size cutoff and the number of groups. To adapt the full pipeline to another country, replace the download and census-parsing steps (`01_download_data.py`, `04_units_census.py`) and keep steps 03 and 05–08.

## Reproducing the analysis

Step 01 downloads the census microdata from INEC. The 2020 OpenStreetMap street network (`data/network/`) is in the data deposit (see Data below). All parameters are in `config.yaml`.

```bash
uv run python pipeline/01_download_data.py
uv run python pipeline/02_country_outline.py
uv run python pipeline/03_percolation.py
uv run python pipeline/04_units_census.py 2010
uv run python pipeline/04_units_census.py 2022
uv run python pipeline/05_build_tree.py
uv run python pipeline/05b_unit_labels_2022geo.py
uv run python pipeline/06_gjsd.py units_2010 2010 unit_labels_sec2022.parquet
uv run python pipeline/06_gjsd.py units_2022 2022 unit_labels_sec2022.parquet
uv run python pipeline/06b_longitudinal.py
uv run python pipeline/07_robustness.py units_2010 2010
uv run python pipeline/07_robustness.py units_2022 2022
uv run python pipeline/08_figures.py
```

The first run downloads about 4 GB from INEC. Percolation takes about 12 s, and the 2010 pipeline excluding robustness takes about 2 minutes on the desktop described above.

## Data

Census microdata © INEC Ecuador (open data), downloaded directly from INEC with recorded checksums. Street-network data © OpenStreetMap contributors (ODbL). The street network and the intermediate and processed datasets are on Zenodo at [doi:10.5281/zenodo.23456682](https://doi.org/10.5281/zenodo.23456682). Unzip the archives into `data/`.

## Citation

Neira, M., Marin, V. & Arcaute, E. The multiscale structure of socio-spatial segregation. Manuscript under review.

Code: [doi:10.5281/zenodo.23457240](https://doi.org/10.5281/zenodo.23457240)

## License

MIT (code).
