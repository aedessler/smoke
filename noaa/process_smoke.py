#!/usr/bin/env python3
"""Process NOAA HMS smoke polygons into CONUS smoke area and gridded day counts.

Which density class the focus products cover is set by SMOKE_CLASS below
("Light", "Medium" or "Heavy") -- change that one constant to retarget the run.

Source: https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/
        Shapefile/Annual_Bundles/hms_smokeYYYY.zip  (one shapefile per year)

Products written to data/processed/ (<cls> is SMOKE_CLASS, lowercased):
  daily_smoke_area.csv          per-UTC-day CONUS area under light/medium/heavy (km^2)
  monthly_<cls>_smoke_area.csv  monthly sum of daily <cls> area (km^2 * days)
  <cls>_smoke_days.nc           <cls>_smoke_days(year, lat, lon) on a 0.1 deg grid

Three properties of this dataset drive the processing choices here, all verified
against the archive rather than assumed:

1. `Density` is the string "Light"/"Medium"/"Heavy", or "NA" where the analyst did not
   classify the plume. It is ~100% "NA" in 2005-2007 and 2009, and 22-27% "NA" in 2008
   and 2010. Density-based products therefore start in 2011. All-density area is still
   recorded from 2006 so the longer record is available.

2. The density classes are strictly NESTED contours, not disjoint regions: on any given
   day Heavy is contained in Medium is contained in Light (verified to within 0.1%).
   So each class means "at least this dense". Note what that implies for SMOKE_CLASS =
   "Light": Light is the OUTER envelope, so the light-smoke products are effectively
   total smoke extent (any density), not a light-only band. Compare `light_km2` with
   `all_km2` in the daily CSV -- they agree to within rounding on most days.

3. Polygon size is extremely long-tailed. The median Heavy polygon is ~1,000-5,000 km2,
   but analysts also draw single continental-scale plumes up to ~11,000,000 km2 (larger
   than CONUS). These giants are genuine hand-drawn analyses with 100-350 vertices, not
   degenerate geometry, so they are NOT filtered out. But one polygon can dominate a
   day's total, so `n_<cls>_polys` and `<cls>_max_poly_km2` are recorded per day to make
   that visible and let downstream users run their own sensitivity tests.

Run: python3 process_smoke.py
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- configuration

BASE_URL = (
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/"
    "Shapefile/Annual_Bundles"
)

AREA_START_YEAR = 2006  # all-density area series (2005 is a partial year: starts Aug 5)
DENSITY_START_YEAR = 2011  # first year with fully populated Density
END_YEAR = 2025  # last complete calendar year

AREA_CRS = "EPSG:5070"  # NAD83 / CONUS Albers Equal Area

# 0.1 degree CONUS grid
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, DGRID = -125.0, -66.0, 24.0, 50.0, 0.1

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

DENSITY_CLASSES = ("Light", "Medium", "Heavy")

# Which class the focus products (monthly series + gridded day counts) cover.
# See note 2 in the module docstring before setting this to "Light".
SMOKE_CLASS = "Light"
CLS = SMOKE_CLASS.lower()


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- download

def download_bundles(years) -> None:
    """Fetch the annual shapefile bundles into data/raw/, skipping ones already there."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        dest = RAW_DIR / f"hms_smoke{year}.zip"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/hms_smoke{year}.zip"
        log(f"  downloading {year} ...")
        for attempt in (1, 2, 3):
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    tmp.rename(dest)
                break
            except Exception as exc:  # noqa: BLE001
                log(f"    attempt {attempt} failed: {exc}")
                if attempt == 3:
                    raise


# ---------------------------------------------------------------- domain + grid

def build_conus():
    """Return (conus_4326, conus_5070) as single dissolved geometries."""
    path = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_1_states_provinces_lakes"
    )
    states = gpd.read_file(path)
    conus = states[
        (states["admin"] == "United States of America")
        & (~states["name"].isin(["Alaska", "Hawaii"]))
    ]
    geom_4326 = shapely.union_all(shapely.make_valid(conus.geometry.values))
    geom_5070 = (
        gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(AREA_CRS).iloc[0]
    )
    return geom_4326, geom_5070


def build_grid(conus_4326):
    """0.1 deg cell centers, an STRtree over them, and an in-CONUS mask."""
    lon = np.arange(LON_MIN + DGRID / 2, LON_MAX, DGRID)
    lat = np.arange(LAT_MIN + DGRID / 2, LAT_MAX, DGRID)
    lon2d, lat2d = np.meshgrid(lon, lat)  # (nlat, nlon)
    points = shapely.points(lon2d.ravel(), lat2d.ravel())
    tree = shapely.STRtree(points)
    in_conus = shapely.contains(conus_4326, points).reshape(lat2d.shape)
    log(f"  grid {lat.size} x {lon.size}, {int(in_conus.sum())} cells inside CONUS")
    return lon, lat, points, tree, in_conus


# ---------------------------------------------------------------- per-year work

def read_year(year: int) -> gpd.GeoDataFrame:
    """Read one annual bundle, repair geometry, and attach a UTC calendar date."""
    # Some years (e.g. 2015) contain rings that are not closed, which is a hard read
    # error otherwise. "ignore" yields None for those; they are dropped and counted below.
    # ("fix" would be preferable but needs GEOS >= 3.11.)
    gdf = gpd.read_file(
        f"zip://{RAW_DIR / f'hms_smoke{year}.zip'}", on_invalid="ignore"
    )
    n_unreadable = int(gdf.geometry.isna().sum())
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # "Start" looks like "2020001 1546" -> YYYY + day-of-year + HHMM (UTC)
    start = pd.to_datetime(gdf["Start"], format="%Y%j %H%M", errors="coerce", utc=True)
    n_bad = int(start.isna().sum())
    gdf = gdf.assign(date=start.dt.date)

    # HMS geometries are hand-drawn; self-intersections abort union_all if left alone.
    n_invalid = int((~gdf.geometry.is_valid).sum())
    if n_invalid:
        gdf.geometry = shapely.make_valid(gdf.geometry.values)

    n_total = len(gdf)
    gdf = gdf[gdf["date"].notna()]
    # Bundles occasionally carry a few stragglers from the adjacent year.
    gdf = gdf[pd.Index([d.year for d in gdf["date"]]) == year]
    gdf["Density"] = gdf["Density"].astype(str).str.strip().str.title()

    log(
        f"  {year}: {n_total} features "
        f"({n_unreadable} unreadable geoms, {n_bad} unparseable dates, "
        f"{n_invalid} repaired geoms, {n_total - len(gdf)} dropped), "
        f"density={dict(gdf['Density'].value_counts())}"
    )
    return gdf


def clip_to_conus(union_4326, conus_5070):
    """Reproject a lat/lon geometry to the equal-area CRS and clip it to CONUS.

    Reprojection of these large hand-drawn polygons can itself produce invalid
    geometry (side-location conflicts), so repair after the transform and fall back
    to a zero-width buffer if the intersection still fails.
    """
    geom = gpd.GeoSeries([union_4326], crs="EPSG:4326").to_crs(AREA_CRS).iloc[0]
    geom = shapely.make_valid(geom)
    try:
        return geom.intersection(conus_5070)
    except shapely.errors.GEOSException:
        return geom.buffer(0).intersection(conus_5070)


def process_year(year, gdf, conus_4326, conus_5070, tree, in_conus, use_density):
    """Return (list of daily area records, SMOKE_CLASS-smoke-day counts for the year)."""
    class_days = np.zeros(in_conus.shape, dtype=np.int16)
    records = []

    for date, day in gdf.groupby("date", sort=True):
        rec = {"date": date}

        # All densities combined, for the long-record context series.
        subsets = {"all": day}
        if use_density:
            for cls in DENSITY_CLASSES:
                subsets[cls.lower()] = day[day["Density"] == cls]

        for label, sub in subsets.items():
            if len(sub) == 0:
                rec[f"{label}_km2"] = 0.0
                if label == CLS:
                    rec[f"n_{CLS}_polys"] = 0
                    rec[f"{CLS}_max_poly_km2"] = 0.0
                continue

            if label == CLS:
                # Diagnostics: a single continental-scale polygon can dominate the day.
                areas = (
                    gpd.GeoSeries(sub.geometry.values, crs="EPSG:4326")
                    .to_crs(AREA_CRS)
                    .area
                    / 1e6
                )
                rec[f"n_{CLS}_polys"] = len(sub)
                rec[f"{CLS}_max_poly_km2"] = float(areas.max())

            # Dissolve so overlapping satellite analyses of one plume count once.
            union = shapely.union_all(sub.geometry.values)
            clipped = clip_to_conus(union, conus_5070)
            rec[f"{label}_km2"] = clipped.area / 1e6

            if label == CLS and not clipped.is_empty:
                hit = tree.query(union, predicate="intersects")
                if hit.size:
                    mask = np.zeros(in_conus.size, dtype=bool)
                    mask[hit] = True
                    class_days += (mask.reshape(in_conus.shape) & in_conus).astype(
                        np.int16
                    )

        records.append(rec)

    n_class_days = sum(1 for r in records if r.get(f"{CLS}_km2", 0) > 0)
    log(
        f"    -> {len(records)} days with smoke, {n_class_days} with {CLS} smoke "
        f"over CONUS, max {CLS}-days/cell = {int(class_days.max())}"
    )
    return records, class_days


# ---------------------------------------------------------------- main

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = list(range(AREA_START_YEAR, END_YEAR + 1))
    density_years = [y for y in years if y >= DENSITY_START_YEAR]

    log("Downloading annual bundles ...")
    download_bundles(years)

    log("Building CONUS domain ...")
    conus_4326, conus_5070 = build_conus()
    log(f"  CONUS area = {conus_5070.area / 1e6:,.0f} km2")
    lon, lat, _points, tree, in_conus = build_grid(conus_4326)

    all_records = []
    grids = {}

    log("Processing years ...")
    for year in years:
        use_density = year >= DENSITY_START_YEAR
        gdf = read_year(year)
        recs, class_days = process_year(
            year, gdf, conus_4326, conus_5070, tree, in_conus, use_density
        )
        all_records.extend(recs)
        if use_density:
            grids[year] = class_days

    # ---- daily table
    daily = pd.DataFrame(all_records).sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    cols = [
        "all_km2",
        "light_km2",
        "medium_km2",
        "heavy_km2",
        f"n_{CLS}_polys",
        f"{CLS}_max_poly_km2",
    ]
    for col in cols:
        if col not in daily:
            daily[col] = np.nan
    daily = daily[["date", *cols]]
    daily.to_csv(OUT_DIR / "daily_smoke_area.csv", index=False)
    log(f"Wrote daily_smoke_area.csv ({len(daily)} rows)")

    # ---- monthly table: complete year x month grid, zero-filled for smoke-free months
    dens = daily[daily["date"].dt.year >= DENSITY_START_YEAR].copy()
    dens["year"] = dens["date"].dt.year
    dens["month"] = dens["date"].dt.month
    grouped = dens.groupby(["year", "month"]).agg(
        **{
            f"{CLS}_km2_days": (f"{CLS}_km2", "sum"),
            f"n_days_with_{CLS}": (f"{CLS}_km2", lambda s: int((s > 0).sum())),
            "all_km2_days": ("all_km2", "sum"),
        }
    )
    full_index = pd.MultiIndex.from_product(
        [density_years, range(1, 13)], names=["year", "month"]
    )
    monthly = grouped.reindex(full_index, fill_value=0).reset_index()
    days_in_month = [
        pd.Period(year=int(r.year), month=int(r.month), freq="M").days_in_month
        for r in monthly.itertuples()
    ]
    monthly[f"mean_daily_{CLS}_km2"] = monthly[f"{CLS}_km2_days"] / days_in_month
    monthly_name = f"monthly_{CLS}_smoke_area.csv"
    monthly.to_csv(OUT_DIR / monthly_name, index=False)
    log(f"Wrote {monthly_name} ({len(monthly)} rows)")

    # ---- gridded annual SMOKE_CLASS-smoke days
    stack = np.stack([grids[y] for y in density_years]).astype(np.int16)
    ds = xr.Dataset(
        {
            f"{CLS}_smoke_days": (
                ("year", "lat", "lon"),
                stack,
                {
                    "long_name": f"days per year with {CLS} smoke overhead",
                    # NOT "days": a time-unit string makes xarray decode this
                    # count variable into timedelta64 on read.
                    "units": "count",
                    "description": (
                        "Count of UTC days on which a HMS smoke polygon with "
                        f"Density='{SMOKE_CLASS}' covered the grid cell center."
                    ),
                },
            ),
            # Needed downstream to tell "no smoke" (a real zero) apart from
            # "outside the domain" (not evaluated).
            "in_conus": (
                ("lat", "lon"),
                in_conus,
                {
                    "long_name": "grid cell center lies within CONUS land",
                    "units": "boolean",
                },
            ),
        },
        coords={
            "year": density_years,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": f"NOAA HMS {CLS} smoke days over CONUS",
            "source": f"{BASE_URL}/hms_smokeYYYY.zip",
            "density_threshold": f"Density == '{SMOKE_CLASS}'",
            "grid": f"{DGRID} deg regular lat/lon, cell centers; masked to CONUS land",
            "area_crs": AREA_CRS,
            "density_note": (
                "Density is ~100% unpopulated in 2005-2007 and 2009 and 22-27% "
                f"unpopulated in 2008 and 2010, so this record starts in {DENSITY_START_YEAR}."
            ),
            "nesting_note": (
                "HMS density classes are nested contours: Heavy is contained in Medium "
                f"is contained in Light, so '{SMOKE_CLASS}' means 'at least "
                f"{CLS}'. Light is the outer envelope, so light-class products are "
                "effectively total smoke extent at any density."
            ),
            "polygon_size_note": (
                "Polygon area is long-tailed: median ~1e3-5e3 km2, but single "
                "analyst-drawn plumes reach ~1e7 km2. No size filtering is applied."
            ),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    nc_name = f"{CLS}_smoke_days.nc"
    ds.to_netcdf(OUT_DIR / nc_name)
    log(f"Wrote {nc_name} {dict(ds.sizes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
