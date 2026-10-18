# Data directory


| Folder | Contents | Provenance |
|---|---|---|
| `network/` | 2020 OpenStreetMap street network of Ecuador: nodes (`ec_nodes.csv`), edge list with lengths (`ec_edgelist.csv`), edge geometries (`ec_edges.csv`) | [doi:10.5281/zenodo.23456682](https://doi.org/10.5281/zenodo.23456682) |
| `raw/` | INEC downloads: 2022 manzana microdata, 2010 microdata harmonized to 2022 geography, variable dictionary, national geodatabase | `pipeline/01_download_data.py` |
| `interim/` | Intermediate artifacts (parsed microdata, ICV per dwelling/manzana, percolation memberships) | pipeline steps 02–05 |
| `processed/` | Final analysis tables (tree with GJSD per node per census year, Υ per city, robustness results) | pipeline steps 06–07 |

To rebuild everything from scratch, run the pipeline steps listed in the top-level README (downloads ~4 GB from INEC on first run).
