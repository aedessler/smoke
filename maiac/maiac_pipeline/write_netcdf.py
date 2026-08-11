"""Assemble and write the monthly checkpoint NetCDF, plus its validation."""
from __future__ import annotations

import datetime as dt
import logging
import os

import numpy as np

from . import config

log = logging.getLogger("maiac.netcdf")

DATA_VARS = (
    "mean_aod_055",
    "mean_smoke_aod_055",
    "smoke_aod_index",
    "smoke_pixel_day_fraction",
    "valid_pixel_day_weight",
    "smoke_pixel_day_weight",
)

VAR_ATTRS = {
    "mean_aod_055": {
        "long_name": "Coverage-weighted mean AOD at 550 nm",
        "units": "1",
        "comment": "sum(daily_aod) / sum(valid_pixel_days) over the month",
    },
    "mean_smoke_aod_055": {
        "long_name": "Mean AOD at 550 nm under the smoke aerosol model",
        "units": "1",
        "comment": (
            "Conditional on the smoke model being selected. Still total-column "
            "AOD, not a smoke-only retrieval."
        ),
    },
    "smoke_aod_index": {
        "long_name": "Smoke-model AOD summed over all valid pixel-days",
        "units": "1",
        "comment": (
            "sum(smoke AOD) / sum(valid pixel-days). Combines smoke frequency "
            "and smoke intensity in one number; goes to zero where smoke is "
            "rare, unlike mean_smoke_aod_055."
        ),
    },
    "smoke_pixel_day_fraction": {
        "long_name": "Fraction of valid pixel-days flagged smoke",
        "units": "1",
        "valid_range": (0.0, 1.0),
    },
    "valid_pixel_day_weight": {
        "long_name": "Count of valid native pixel-days aggregated into the cell",
        "units": "1",
        "comment": (
            "The denominator of every ratio above. Low values mean a thin "
            "sample -- read the ratios with this in hand."
        ),
    },
    "smoke_pixel_day_weight": {
        "long_name": "Count of smoke-model native pixel-days in the cell",
        "units": "1",
    },
}


def build_dataset(fields: dict[str, np.ndarray], month: str, extra_attrs: dict) -> "xr.Dataset":
    import xarray as xr

    x, y = config.target_coords()
    year, mon = (int(p) for p in month.split("-"))
    time = np.array([dt.datetime(year, mon, 1)], dtype="datetime64[ns]")

    data = {
        name: (("time", "y", "x"), fields[name][None, :, :].astype("float32"))
        for name in DATA_VARS
    }
    ds = xr.Dataset(
        data,
        coords={"time": time, "y": y, "x": x},
    )

    for name, attrs in VAR_ATTRS.items():
        ds[name].attrs.update(attrs)
        ds[name].attrs["grid_mapping"] = "crs"

    ds["x"].attrs.update(
        standard_name="projection_x_coordinate", units="m", long_name="Easting"
    )
    ds["y"].attrs.update(
        standard_name="projection_y_coordinate", units="m", long_name="Northing"
    )
    ds["time"].attrs.update(
        long_name="First day of the represented month",
        comment="Monthly aggregate; the timestamp labels the month, not an instant.",
    )

    ds["crs"] = np.int32(0)
    ds["crs"].attrs.update(
        grid_mapping_name="albers_conical_equal_area",
        standard_parallel=(29.5, 45.5),
        longitude_of_central_meridian=-96.0,
        latitude_of_projection_origin=23.0,
        false_easting=0.0,
        false_northing=0.0,
        spatial_ref=config.TARGET_CRS,
        crs_wkt=config.TARGET_CRS,
    )

    ds.attrs.update(
        title="Monthly 25 km CONUS MAIAC aerosol optical depth and smoke statistics",
        source_product=f"{config.SHORT_NAME}.{config.VERSION}",
        wavelength="550 nm",
        native_resolution="approximately 1 km (MODIS sinusoidal)",
        output_resolution="25 km",
        output_crs=config.TARGET_CRS,
        surface_class="land (AOD_QA bits 3-4 == 0)",
        smoke_aerosol_model="AOD_QA bits 13-14 == 1",
        aggregation=(
            "observation -> native pixel-day -> 25 km monthly ratio of sums"
        ),
        conus_mask="applied at native 1 km resolution, before spatial aggregation",
        Conventions="CF-1.10",
        history=(
            f"created {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
            "by maiac_pipeline"
        ),
        **extra_attrs,
    )
    return ds


def write(ds, path: str) -> str:
    """Compressed NETCDF4, written atomically."""
    encoding = {
        name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": -9999.0,
        }
        for name in DATA_VARS
    }
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=encoding)
    os.replace(tmp, path)
    return path


def validate(ds) -> list[str]:
    """Plan section 23 checks. Returns a list of human-readable problems.

    Deliberately returns rather than raises, so the caller can log every
    problem at once instead of surfacing them one restart at a time.
    """
    problems = []

    def finite(name):
        return ds[name].values[np.isfinite(ds[name].values)]

    for name in ("mean_aod_055", "mean_smoke_aod_055"):
        vals = finite(name)
        if vals.size and vals.min() < 0:
            problems.append(f"{name} has negative values (min {vals.min():.4g})")

    frac = finite("smoke_pixel_day_fraction")
    if frac.size and (frac.min() < 0 or frac.max() > 1 + 1e-6):
        problems.append(
            f"smoke_pixel_day_fraction outside [0, 1] "
            f"({frac.min():.4g} .. {frac.max():.4g})"
        )

    valid_w = ds["valid_pixel_day_weight"].values
    smoke_w = ds["smoke_pixel_day_weight"].values
    over = np.nansum(smoke_w > valid_w + 1e-6)
    if over:
        problems.append(f"smoke_pixel_day_weight exceeds valid weight in {over} cells")

    x, y = config.target_coords()
    if not np.allclose(ds["x"].values, x):
        problems.append("x coordinates do not match the frozen reference grid")
    if not np.allclose(ds["y"].values, y):
        problems.append("y coordinates do not match the frozen reference grid")
    if ds.sizes["y"] != config.TARGET_NY or ds.sizes["x"] != config.TARGET_NX:
        problems.append(
            f"grid is {ds.sizes['y']}x{ds.sizes['x']}, "
            f"expected {config.TARGET_NY}x{config.TARGET_NX}"
        )

    t = ds["time"].values[0].astype("datetime64[D]").astype(object)
    if t.day != 1:
        problems.append(f"time {t} is not the first day of a month")

    if float(np.nansum(valid_w)) <= 0:
        problems.append("no valid pixel-days anywhere in the month")

    return problems


def diagnostics(ds) -> dict:
    """Numbers worth having in the run log for every month."""
    valid_w = ds["valid_pixel_day_weight"].values
    covered = int(np.sum(valid_w > 0))
    aod = ds["mean_aod_055"].values
    smoke_frac = ds["smoke_pixel_day_fraction"].values
    return {
        "cells_with_data": covered,
        "cells_total": config.TARGET_NCELLS,
        "coverage_pct": round(100.0 * covered / config.TARGET_NCELLS, 1),
        "mean_aod_median": _r(np.nanmedian(aod)),
        "mean_aod_p99": _r(np.nanpercentile(aod, 99)) if covered else None,
        "smoke_fraction_mean": _r(np.nanmean(smoke_frac)),
        "smoke_fraction_max": _r(np.nanmax(smoke_frac)) if covered else None,
        "total_valid_pixel_days": float(np.nansum(valid_w)),
    }


def _r(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 5)
