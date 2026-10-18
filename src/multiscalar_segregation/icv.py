"""Index of Life Conditions (ICV) from INEC census microdata.

The index is a weighted sum of component scores, with weights matching the
dimension structure of the SI table:

    ICV = 1/24 (floor + walls + roof)                     housing quality
        + 1/20 (water + drainage) + 1/40 waste            water & sanitation
        + 1/32 (electricity + rooms + bedrooms_rs
                + kitchen + bathroom + stove)             services & space
        + 1/64 (phone + cell + internet + tv)             telecommunications
        + 1/4 schooling                                   education
        + 1/4 healthcare                                  health coverage
"""

from __future__ import annotations

import pandas as pd

ICV_WEIGHTS: dict[str, float] = {
    "floor": 1 / 24, "walls": 1 / 24, "roof": 1 / 24,
    "water": 1 / 20, "drainage": 1 / 20, "waste": 1 / 40,
    "electricity": 1 / 32, "rooms": 1 / 32, "bedrooms_rs": 1 / 32,
    "kitchen": 1 / 32, "bathroom": 1 / 32, "stove": 1 / 32,
    "phone": 1 / 64, "cell": 1 / 64, "internet": 1 / 64, "tv": 1 / 64,
    "schooling": 1 / 4, "healthcare": 1 / 4,
}


def icv_from_components(components: pd.DataFrame) -> pd.Series:
    """Compose the ICV from component scores using the exact recovered weights."""
    missing = set(ICV_WEIGHTS) - set(components.columns)
    if missing:
        raise ValueError(f"missing ICV components: {sorted(missing)}")
    icv = sum(w * components[c] for c, w in ICV_WEIGHTS.items())
    return icv.rename("ICV")


def schooling_score(escola: pd.Series, age: pd.Series) -> pd.Series:
    """Per-person schooling score (exact 2010 formula, applied to both years).

    s = 0.5 + 0.5 * ESCOLA / min(13, age - 5); 0 if ESCOLA == 0 or age <= 5;
    NaN (excluded) if ESCOLA is missing or the 99 sentinel.
    """
    import numpy as np

    e = pd.to_numeric(escola, errors="coerce").astype(np.float64)
    a = pd.to_numeric(age, errors="coerce").astype(np.float64)
    denom = np.minimum(13.0, a - 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 0.5 + 0.5 * e / denom
    s = s.where((e != 0) & (denom > 0), 0.0)
    return s.where(e.notna() & (e != 99), other=np.nan)


def healthcare_score(covered: pd.Series, in_universe: pd.Series) -> pd.Series:
    """Per-person health score: 1.0 if covered else 0.5; NaN outside the universe
    (2010: age >= 12, operationalised as non-missing affiliation answer)."""
    import numpy as np

    s = pd.Series(0.5, index=covered.index).where(~covered, 1.0)
    return s.where(in_universe, other=np.nan)


def bedrooms_rescale(bedrooms: pd.Series, province: pd.Series) -> pd.Series:
    """Province-anchored rescale of bedrooms-per-person (exact 2010 rule):
    0 if bedrooms == 0, else 1 + (bedrooms - 1) / K_province, with
    K_province = max(bedrooms) over the province's dwellings - 1."""
    k = bedrooms.groupby(province).transform("max") - 1.0
    rs = 1.0 + (bedrooms - 1.0) / k
    return rs.where(bedrooms != 0, 0.0)


def mean_of_household_means(person_scores: pd.Series, household_id: pd.Series,
                            dwelling_id: pd.Series) -> pd.Series:
    """Aggregate person scores to dwellings: mean per household (skipping NaN),
    then mean of household means per dwelling."""
    df = pd.DataFrame({"s": person_scores, "hog": household_id, "viv": dwelling_id})
    hh = df.groupby(["viv", "hog"])["s"].mean()
    return hh.groupby(level="viv").mean()
