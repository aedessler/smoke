#!/usr/bin/env python3
"""Grid all-density smoke days, the one product process_smoke.py does not write.

process_smoke.py grids only the class named by its SMOKE_CLASS, so the maps have
always been density-class maps starting in 2011. This adds the all-density field
on the same grid, which needs no Density column and so covers the full 2006-2025
record -- five extra years, including the four (2006, 2007, 2009, and partly 2008
and 2010) where Density is unpopulated and the class products cannot exist at all.

Cheaper than the equivalent loop in process_smoke.py because it skips the union:
that script needs `union_all` per day for the area series anyway, but a day count
does not -- a cell is covered if ANY polygon covers it, so the polygons go into
the STRtree query directly. Whole record in about a minute rather than ~20.

Reads data/raw/hms_smokeYYYY.zip, writes data/processed/all_smoke_days.nc with the
same layout as <cls>_smoke_days.nc, so plot_smoke.py --class all can map it.

Run: python3 grid_all_smoke.py
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import xarray as xr

from process_smoke import (
    AREA_CRS,
    AREA_START_YEAR,
    BASE_URL,
    DGRID,
    END_YEAR,
    OUT_DIR,
    build_conus,
    build_grid,
    download_bundles,
    log,
    read_year,
)


def grid_year(gdf, tree, in_conus) -> np.ndarray:
    """Days per cell on which any HMS smoke polygon covered the cell center."""
    counts = np.zeros(in_conus.shape, dtype=np.int16)
    for _date, day in gdf.groupby("date", sort=True):
        # Query every polygon of the day at once; row 1 of the result indexes the
        # grid points. Duplicates across overlapping polygons collapse in the mask,
        # so a cell is still counted once per day.
        hit = tree.query(day.geometry.values, predicate="intersects")
        if hit.size:
            mask = np.zeros(in_conus.size, dtype=bool)
            mask[hit[1]] = True
            counts += (mask.reshape(in_conus.shape) & in_conus).astype(np.int16)
    return counts


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = list(range(AREA_START_YEAR, END_YEAR + 1))

    log("Checking annual bundles ...")
    download_bundles(years)

    log("Building CONUS domain ...")
    conus_4326, _conus_5070 = build_conus()
    lon, lat, _points, tree, in_conus = build_grid(conus_4326)

    log("Gridding years ...")
    grids = {}
    for year in years:
        grids[year] = grid_year(read_year(year), tree, in_conus)
        log(f"    -> max smoke-days/cell = {int(grids[year].max())}")

    stack = np.stack([grids[y] for y in years]).astype(np.int16)
    ds = xr.Dataset(
        {
            "all_smoke_days": (
                ("year", "lat", "lon"),
                stack,
                {
                    "long_name": "days per year with smoke of any density overhead",
                    # NOT "days": a time-unit string makes xarray decode this
                    # count variable into timedelta64 on read.
                    "units": "count",
                    "description": (
                        "Count of UTC days on which any HMS smoke polygon, of any "
                        "Density value including unclassified, covered the cell center."
                    ),
                },
            ),
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
            "year": years,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "NOAA HMS smoke days (any density) over CONUS",
            "source": f"{BASE_URL}/hms_smokeYYYY.zip",
            "density_threshold": "none -- every polygon counts",
            "grid": f"{DGRID} deg regular lat/lon, cell centers; masked to CONUS land",
            "area_crs": AREA_CRS,
            "record_note": (
                "No Density filtering, so this record starts in "
                f"{AREA_START_YEAR} rather than 2011 like the class products. "
                "2005 is excluded: the HMS archive begins 2005-08-05."
            ),
            "nesting_note": (
                "HMS density classes are nested contours, and Light is the outer "
                "envelope, so over 2011-2025 this field is near-identical to "
                "light_smoke_days. The 2006-2010 years are the reason to use it."
            ),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    ds.to_netcdf(OUT_DIR / "all_smoke_days.nc")
    log(f"Wrote all_smoke_days.nc {dict(ds.sizes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
