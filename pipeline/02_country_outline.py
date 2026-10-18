"""Step 02: dissolve the census sectors into a national outline for map figures.
"""

import geopandas as gpd
import pyogrio
from shapely.geometry import MultiPolygon

from multiscalar_segregation import load_config, repo_root

CFG = load_config()
ROOT = repo_root()
GPKG = ROOT / "data/raw/geodatabase/GEODATABASE_NACIONAL_2021/GEODATABASE_NACIONAL_2021.gpkg"
OUT = ROOT / CFG["paths"]["data_interim"] / "ecuador_outline.geojson"


def main() -> None:
    print("reading sec_a geometries...", flush=True)
    g = pyogrio.read_dataframe(GPKG, layer="sec_a", columns=[])
    print(f"{len(g)} polygons, dissolving...", flush=True)
    geom = g.geometry.buffer(50).union_all().buffer(-50)
    geom = geom.simplify(150)  # metres (UTM 17S)
    out = gpd.GeoDataFrame(geometry=[geom], crs=g.crs).to_crs("EPSG:4326")
    parts = out.geometry.iloc[0]
    polys = list(parts.geoms) if parts.geom_type == "MultiPolygon" else [parts]
    areas = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs("EPSG:32717").area
    big = [p for p, a in zip(polys, areas) if a > 2e6]
    print(f"{len(polys)} parts -> {len(big)} kept", flush=True)
    gpd.GeoDataFrame(geometry=[MultiPolygon(big)], crs="EPSG:4326") \
        .to_file(OUT, driver="GeoJSON")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
