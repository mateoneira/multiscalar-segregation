"""Step 04: dwelling-level ICV and census-unit group counts, per census year.

Usage: 04_units_census.py 2010 | 2022

Implements the ICV component definitions for the 2010 census and their
harmonized 2022 equivalents (multiscalar_segregation.icv_maps).

Harmonized ICV (both years, used for the longitudinal analysis):
- healthcare = social-security affiliation only (the 2022 census dropped the
  private-insurance question);
- bedrooms-per-person from the household bedroom counts (sum of H01; the 2010
  dwelling-level TOTDOR total does not exist in 2022).
For 2010 the unharmonized index (with private insurance and TOTDOR) is also
computed as ICV_orig.

Census unit: 2022 census sector = first 12 digits of the dwelling id (identical
for both years. The 2010 microdata is the release harmonized to 2022
geography).
"""

import sys

import numpy as np
import pandas as pd

from multiscalar_segregation import (geography, icv, icv_maps, load_config, metrics,
                                     repo_root)

RAW_2010 = "data/raw/cpv2010h/1.2 BDD_CPV_2010_MANLOC_CON_CORRESP_2022_CSV"
RAW_2022 = "data/raw/cpv2022/1.3 BDD_CPV_2022_MANLOC_CSV"
CHUNK = 2_000_000


def read_scored(path, id_col, maps, extra_numeric=(), extra_str=()):
    """Read a census table, mapping categorical answers to scores on the fly."""
    sources = [v for v, _ in maps.values()]
    usecols = [id_col, *sources, *extra_numeric, *extra_str]
    df = pd.read_csv(path, sep=";", usecols=usecols, dtype=str)
    out = pd.DataFrame(index=df.index)
    out["id"] = pd.to_numeric(df[id_col], errors="coerce").astype(np.int64)
    for comp, (var, mp) in maps.items():
        out[comp] = df[var].map(mp).astype(np.float64)
    for c in extra_numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    for c in extra_str:
        out[c] = df[c].astype("string[pyarrow]")
    return out


def aggregate_persons(pob_path, usecols, score_fns: dict) -> pd.DataFrame:
    """Chunked person -> household aggregation (mean per score + person count).

    Households are keyed by (dwelling int64, household ordinal int16), derived
    from the last two digits of ID_HOG.
    """
    running = None
    done = 0
    for chunk in pd.read_csv(pob_path, sep=";", usecols=usecols + ["ID_HOG"],
                             dtype=str, chunksize=CHUNK):
        keys = pd.DataFrame({
            "viv": pd.to_numeric(chunk["ID_HOG"].str.slice(0, -2),
                                 errors="coerce").astype(np.int64),
            "hog": pd.to_numeric(chunk["ID_HOG"].str.slice(-2),
                                 errors="coerce").astype(np.int16),
        })
        for name, fn in score_fns.items():
            s = fn(chunk)
            keys[f"{name}_sum"] = s.fillna(0.0).to_numpy()
            keys[f"{name}_n"] = s.notna().to_numpy().astype(np.int32)
        keys["persons"] = 1
        agg = keys.groupby(["viv", "hog"]).sum()
        running = agg if running is None else (
            pd.concat([running, agg]).groupby(level=[0, 1]).sum())
        done += len(chunk)
        print(f"  persons processed: {done:,}", flush=True)
    out = pd.DataFrame(index=running.index)
    for name in score_fns:
        out[name] = running[f"{name}_sum"] / running[f"{name}_n"].replace(0, np.nan)
    out["persons"] = running["persons"]
    return out


def household_means_to_dwellings(hh: pd.DataFrame, score_cols) -> pd.DataFrame:
    """Mean of household means per dwelling + total person count."""
    g = hh.groupby(level="viv")
    out = g[list(score_cols)].mean()
    out["num_people"] = g["persons"].sum()
    out["num_homes_pob"] = g.size()
    return out


def persons_stage(year: str, out_dir) -> None:
    """Heavy stage: chunked person->household aggregation, written to parquet."""
    root = repo_root()
    if year == "2010":
        def schooling_fn(c):
            return icv.schooling_score(c["ESCOLA"], c["P03"])

        def health_orig_fn(c):
            p35 = c["P35"]
            covered = p35.isin(list("123456")).fillna(False).astype(bool) | \
                (c["P07"] == "1").fillna(False).astype(bool)
            return icv.healthcare_score(pd.Series(np.asarray(covered), index=c.index),
                                        pd.Series(p35.notna().to_numpy(), index=c.index))

        def health_harm_fn(c):
            p35 = c["P35"]
            covered = p35.isin(list("123456")).fillna(False).astype(bool)
            return icv.healthcare_score(pd.Series(np.asarray(covered), index=c.index),
                                        pd.Series(p35.notna().to_numpy(), index=c.index))

        hh = aggregate_persons(root / RAW_2010 / "CPV_2010_Poblacion_Manloc_Corresp_2022.csv",
                               ["ESCOLA", "P03", "P35", "P07"],
                               {"schooling": schooling_fn,
                                "healthcare_orig": health_orig_fn,
                                "healthcare": health_harm_fn})
    else:
        def schooling_fn(c):
            return icv.schooling_score(c["ESCOLA"], c["P03"])

        def health_fn(c):
            p30 = c["P30"]
            covered = p30.isin(sorted(icv_maps.P30_COVERED)).fillna(False).astype(bool)
            return icv.healthcare_score(pd.Series(np.asarray(covered), index=c.index),
                                        pd.Series(p30.notna().to_numpy(), index=c.index))

        hh = aggregate_persons(root / RAW_2022 / "CPV_2022_Poblacion_Manloc.csv",
                               ["ESCOLA", "P03", "P30"],
                               {"schooling": schooling_fn, "healthcare": health_fn})
    hh.to_parquet(out_dir / "hh_agg.parquet")
    print(f"wrote {out_dir / 'hh_agg.parquet'} ({len(hh):,} households)")


def build_2010():
    root = repo_root()
    d = root / RAW_2010

    print("Vivienda 2010...", flush=True)
    viv = read_scored(d / "CPV_2010_Vivienda_Manloc_Corresp_2022.csv", "ID_VIV",
                      icv_maps.MAPS_2010_VIVIENDA,
                      extra_numeric=("V14", "TOTDOR", "TOTPER"),
                      extra_str=("ID_VIV_2010",)).set_index("id")

    print("Hogar 2010...", flush=True)
    hog = read_scored(d / "CPV_2010_Hogar_Manloc_Corresp_2022.csv", "ID_VIV",
                      icv_maps.MAPS_2010_HOGAR, extra_numeric=("H01",),
                      extra_str=("ID_HOG",))
    dw_h = hog.groupby("id").mean(numeric_only=True)
    sum_h01 = hog.groupby("id")["H01"].sum()
    num_homes = hog.groupby("id").size().rename("num_homes")

    print("household aggregates (from persons stage)...", flush=True)
    hh = pd.read_parquet(root / "data/interim/units_2010/hh_agg.parquet")
    valid_hh = pd.MultiIndex.from_arrays(
        [pd.to_numeric(hog["ID_HOG"].str.slice(0, -2), errors="coerce").astype(np.int64),
         pd.to_numeric(hog["ID_HOG"].str.slice(-2), errors="coerce").astype(np.int16)],
        names=["viv", "hog"])
    hh = hh[hh.index.isin(valid_hh)]
    dw_p = household_means_to_dwellings(hh, ["schooling", "healthcare",
                                             "healthcare_orig"])
    del hh
    id_viv_2010 = viv["ID_VIV_2010"]

    print("composing 2010 components...", flush=True)
    comp = viv[list(icv_maps.MAPS_2010_VIVIENDA)].copy()
    province = (viv.index // 10 ** (geography.ID_VIV_DIGITS["2010"] - 2)).astype(np.int64)

    comp["rooms"] = (viv["V14"] > sum_h01.reindex(viv.index)).astype(float) \
        .where(viv["V14"].notna())
    bedrooms_orig = viv["TOTDOR"] / viv["TOTPER"]
    bedrooms_harm = sum_h01.reindex(viv.index) / viv["TOTPER"]
    del viv
    import gc; gc.collect()
    comp["bedrooms"] = bedrooms_harm
    comp["bedrooms_rs"] = icv.bedrooms_rescale(bedrooms_harm, province)
    # the unharmonized index anchors the rescale on the 2010 province: dwellings
    # moved by the 2012-2016 boundary referendums (province 90, La Concordia)
    # keep their original anchor
    province_2010 = id_viv_2010.str.slice(0, 2)
    bedrooms_rs_orig = icv.bedrooms_rescale(bedrooms_orig,
                                            pd.Series(province_2010.to_numpy(),
                                                      index=comp.index))
    for c in icv_maps.MAPS_2010_HOGAR:
        comp[c] = dw_h[c].reindex(comp.index)
    del dw_h, hog
    for c in ["schooling", "healthcare", "healthcare_orig"]:
        comp[c] = dw_p[c].reindex(comp.index)
    gc.collect()

    comp["ICV"] = icv.icv_from_components(comp)
    comp["ICV_orig"] = (comp["ICV"]
                        + icv.ICV_WEIGHTS["healthcare"]
                        * (comp["healthcare_orig"] - comp["healthcare"])
                        + icv.ICV_WEIGHTS["bedrooms_rs"]
                        * (bedrooms_rs_orig - comp["bedrooms_rs"]))
    comp["num_people"] = dw_p["num_people"].reindex(comp.index).fillna(0).astype(int)
    comp["num_homes"] = num_homes.reindex(comp.index).fillna(0).astype(int)
    comp["ID_VIV_2010"] = id_viv_2010
    return comp


def build_2022():
    root = repo_root()
    d = root / RAW_2022

    print("Vivienda 2022...", flush=True)
    viv = read_scored(d / "CPV_2022_Vivienda_Manloc.csv", "ID_VIV",
                      icv_maps.MAPS_2022_VIVIENDA,
                      extra_numeric=("V15", "TOTPER"),
                      extra_str=("V0201", "V12", "V13", "IMP_VOPA")).set_index("id")
    # INEC V0201: 1 = occupied, persons present; 2 = occupied, persons absent
    # (hot-deck imputed, flagged IMP_VOPA == 1)
    viv = viv[viv["V0201"].isin(["1", "2"])]
    viv["electricity"] = np.where(viv["V12"].isna() & viv["V13"].isna(), np.nan,
                                  ((viv["V12"] == "1") | (viv["V13"] == "1"))
                                  .astype(float))

    print("Hogar 2022...", flush=True)
    hog = read_scored(d / "CPV_2022_Hogar_Manloc.csv", "ID_VIV",
                      icv_maps.MAPS_2022_HOGAR, extra_numeric=("H01",),
                      extra_str=("ID_HOG",))
    dw_h = hog.groupby("id").mean(numeric_only=True)
    sum_h01 = hog.groupby("id")["H01"].sum()
    num_homes = hog.groupby("id").size().rename("num_homes")

    print("household aggregates (from persons stage)...", flush=True)
    hh = pd.read_parquet(root / "data/interim/units_2022/hh_agg.parquet")
    valid_hh = pd.MultiIndex.from_arrays(
        [pd.to_numeric(hog["ID_HOG"].str.slice(0, -2), errors="coerce").astype(np.int64),
         pd.to_numeric(hog["ID_HOG"].str.slice(-2), errors="coerce").astype(np.int16)],
        names=["viv", "hog"])
    hh = hh[hh.index.isin(valid_hh)]
    dw_p = household_means_to_dwellings(hh, ["schooling", "healthcare"])
    del hh
    imp_vopa = viv["IMP_VOPA"]

    print("composing 2022 components...", flush=True)
    comp = viv[list(icv_maps.MAPS_2022_VIVIENDA) + ["electricity"]].copy()
    province = (viv.index // 10 ** (geography.ID_VIV_DIGITS["2022"] - 2)).astype(np.int64)

    comp["rooms"] = (viv["V15"] > sum_h01.reindex(viv.index)).astype(float) \
        .where(viv["V15"].notna())
    bedrooms = sum_h01.reindex(viv.index) / viv["TOTPER"]
    del viv
    import gc; gc.collect()
    comp["bedrooms"] = bedrooms
    comp["bedrooms_rs"] = icv.bedrooms_rescale(bedrooms, province)
    for c in icv_maps.MAPS_2022_HOGAR:
        comp[c] = dw_h[c].reindex(comp.index)
    del dw_h, hog
    for c in ["schooling", "healthcare"]:
        comp[c] = dw_p[c].reindex(comp.index)
    gc.collect()

    comp["ICV"] = icv.icv_from_components(comp)
    comp["num_people"] = dw_p["num_people"].reindex(comp.index).fillna(0).astype(int)
    comp["num_homes"] = num_homes.reindex(comp.index).fillna(0).astype(int)
    comp["IMP_VOPA"] = imp_vopa
    return comp


def write_units(table: pd.DataFrame, year: str, n_groups: int, out_dir) -> None:
    """National deciles of the harmonized ICV -> per-sector group counts."""
    ok = table.dropna(subset=["ICV"]).copy()
    print(f"dwellings with ICV: {len(ok):,} / {len(table):,}")
    edges = metrics.quantile_edges(ok["ICV"].to_numpy(), n_groups)
    np.save(out_dir / "decile_edges.npy", edges)
    idx = np.clip(np.searchsorted(edges, ok["ICV"].to_numpy(), side="right") - 1,
                  0, n_groups - 1)
    width = geography.ID_VIV_DIGITS[year]
    ok["group"] = idx
    ok["sector_id"] = (ok.index // 10 ** (width - 12)).astype(np.int64) \
        .astype(str).str.zfill(12)

    def counts_frame(df):
        counts = (df.pivot_table(index="sector_id", columns="group", values="ICV",
                                 aggfunc="count", fill_value=0)
                  .reindex(columns=range(n_groups), fill_value=0))
        counts.columns = [f"g{c}" for c in counts.columns]
        extras = df.groupby("sector_id").agg(
            num_dwellings=("ICV", "size"),
            total_population=("num_people", "sum"),
            mean_icv=("ICV", "mean"))
        return counts.join(extras)

    counts_frame(ok).to_parquet(out_dir / "unit_counts.parquet")
    if "IMP_VOPA" in ok.columns:
        sub = ok[ok["IMP_VOPA"] != "1"]
        counts_frame(sub).to_parquet(out_dir / "unit_counts_noimp.parquet")
        print(f"IMP_VOPA-excluded variant: {len(sub):,} dwellings "
              f"({1 - len(sub)/len(ok):.2%} removed)")


def main(year: str, stage: str = "all") -> None:
    cfg = load_config()
    root = repo_root()
    out = root / cfg["paths"]["data_interim"] / f"units_{year}"
    out.mkdir(parents=True, exist_ok=True)
    if stage in ("persons", "all") and not (out / "hh_agg.parquet").exists():
        persons_stage(year, out)
    if stage in ("compose", "all"):
        table = build_2010() if year == "2010" else build_2022()
        table.to_parquet(out / "dwellings.parquet")
        write_units(table, year, int(cfg["groups"]["n_groups"]), out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
