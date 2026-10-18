# The Multiscale Structure of Socio-Spatial Segregation

Code for *The multiscale structure of socio-spatial segregation* (Neira, Marin & Arcaute).

Most segregation measures work at one fixed scale or follow administrative boundaries, which hides the scales where social divisions actually operate. This repository implements **percolation divergence trees**, a framework that measures segregation across the whole hierarchy of spatial scales in a country:

1. **Percolation on the street network.** Sweeping a distance threshold over the national road network merges settlements into larger and larger clusters. The result is a hierarchical tree of the spatial system, running from neighbourhoods to macro-regions.
2. **Information-theoretic decomposition.** At each level of the tree, the **Generalised Jensen–Shannon Divergence (GJSD)** measures how much information about socio-economic composition is gained by splitting a cluster into its children. The value given by the GJSD at each node can be interpreted as measuring the segregation of the system at the scale. 

We apply the framework to **Ecuador** using a census-derived index of life conditions (ICV) for the **2010 and 2022 censuses**.

## Reproducing the analysis

```bash
uv sync                              # create environment from uv.lock
uv run pytest                        # verify the library
```

The census microdata are downloaded from INEC; the 2020 OpenStreetMap street network (`data/network/`) is deposited with the data (see Data below). All parameters live in `config.yaml`. 

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

## Data

Census microdata © INEC Ecuador (open data), downloaded directly from INEC with recorded checksums. Street-network data © OpenStreetMap contributors (ODbL). Street network, intermediate and processed datasets: [doi:10.5281/zenodo.23456682](https://doi.org/10.5281/zenodo.23456682), unzip the archives into `data/`.

## Citation

Neira, M., Marin, V. & Arcaute, E. The multiscale structure of socio-spatial segregation. Citation and DOI to be added.

## License

MIT (code).
