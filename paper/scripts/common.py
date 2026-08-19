#!/usr/bin/env python3
"""Shared loaders, regridding and design tokens for the preprint figures.

This is the only module in `paper/` that touches the two source archives. Every
figure script imports from here, so the common grid, the CONUS mask and the
smoke-day conversion are defined exactly once and cannot drift between figures.

The two products live in sibling folders and deliberately share no code
(see ../../README.md). `paper/` sits above both and is allowed to read both;
it still imports nothing from either, only their output files.

    ../../noaa/data/processed/{heavy,light}_smoke_days.nc   HMS, 0.1 deg, annual counts
    ../../noaa/data/processed/daily_smoke_area.csv          HMS, CONUS totals, daily
    ../../modis/data/maiac_smoke_25km_monthly.nc            MAIAC, 25 km Albers, monthly
"""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
HMS_DIR = ROOT / "noaa" / "data" / "processed"
MAIAC_NC = ROOT / "modis" / "data" / "maiac_smoke_25km_monthly.nc"
FIG_DIR = PAPER / "figures"
RESULT_DIR = PAPER / "results"
CACHE = PAPER / "results" / ".cache"

# --------------------------------------------------------------------------
# Analysis window
# --------------------------------------------------------------------------

# HMS density is unusable before 2011 (see ../../noaa/README.md), and the MAIAC
# archive ends 2025-07. 2011-2024 is therefore the span of complete calendar
# years available on both sides, and it is what every annual statistic uses.
# Monthly statistics run to 2025-07 because a month is either complete or absent.
FIRST_YEAR = 2011
LAST_FULL_YEAR = 2024
MONTHLY_END = "2025-07"

# A 25 km cell is kept only if this much of it is CONUS land on the HMS grid.
# Border cells are half-empty on one side and not on the other; letting them in
# puts a rim of spurious disagreement around the whole domain.
MIN_CONUS_FRACTION = 0.5

# --------------------------------------------------------------------------
# Design tokens -- copied from ../../modis/maiac/plot_annual_smoke.py so the
# preprint figures sit next to the repository figures without a restyle.
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HMS = "#c2531f"       # HMS, the analyst product
MAIAC = "#2a78d6"     # MAIAC, the retrieval product
HMS_PALE = "#eec4ac"
MAIAC_PALE = "#a9cbf0"

DPI = 200  # print-resolution; the repository figures use 150

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "font.size": 9,
    "pdf.fonttype": 42,
})

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style_axes(ax, grid_axis: str = "y"):
    """Recessive chrome: no top/right spines, hairline grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  wrote figures/{name}")
    return out


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_maiac() -> xr.Dataset:
    return xr.open_dataset(MAIAC_NC)


def load_hms_grid(cls: str = "heavy") -> xr.Dataset:
    """HMS annual smoke-day counts on the native 0.1 deg grid.

    `cls` is "heavy" or "light"; light is the outer envelope of the nested
    density contours, i.e. smoke at any density.
    """
    return xr.open_dataset(HMS_DIR / f"{cls}_smoke_days.nc")


def load_hms_daily() -> pd.DataFrame:
    """Daily CONUS-clipped smoke area, one row per day that had any smoke.

    Reindexed onto every calendar day of 2006-2025 so that smoke-free days are
    explicit zeros. They are genuine zeros, not gaps: `process_smoke.py` reads
    every annual bundle end to end and writes a row only when the dissolved
    area is nonzero.
    """
    df = pd.read_csv(HMS_DIR / "daily_smoke_area.csv", parse_dates=["date"])
    full = pd.date_range(f"{df.date.dt.year.min()}-01-01",
                         f"{df.date.dt.year.max()}-12-31", freq="D")
    df = df.set_index("date").reindex(full)
    df.index.name = "date"
    # all_km2 exists from 2006; the density columns only from 2011, and NaN
    # before that means "not measured", which must not become a zero.
    df["all_km2"] = df["all_km2"].fillna(0.0)
    for c in ("light_km2", "medium_km2", "heavy_km2"):
        known = df.index.year >= FIRST_YEAR
        df.loc[known, c] = df.loc[known, c].fillna(0.0)
    return df.reset_index()


# --------------------------------------------------------------------------
# MAIAC -> smoke days
# --------------------------------------------------------------------------

def maiac_smoke_days(ds: xr.Dataset) -> xr.DataArray:
    """Monthly MAIAC smoke frequency expressed in days, to match HMS units.

        smoke_days = (smoke pixel-days / valid pixel-days) x days in month

    The multiplication extrapolates the clear-sky smoke frequency to every day
    of the month. That is an assumption -- smoke occurrence is not independent
    of cloudiness -- and it is the reason Section 4 reports the MAIAC:HMS ratio
    rather than treating either as truth. It is nonetheless the only conversion
    that puts the two products in the same unit without inventing a threshold.

    Cells with no valid observation in a month contribute 0 rather than NaN:
    "MODIS saw nothing smoky here" and "MODIS saw nothing here" are different
    statements, and the weight field carries the difference for anyone who
    needs it (`valid_pixel_day_weight` is 0 in exactly those cells).
    """
    frac = ds["smoke_pixel_day_fraction"].fillna(0.0)
    ndays = xr.DataArray(
        [calendar.monthrange(int(t.dt.year), int(t.dt.month))[1] for t in ds["time"]],
        coords={"time": ds["time"]}, dims="time",
    )
    out = frac * ndays
    out.name = "maiac_smoke_days"
    out.attrs["units"] = "count"
    return out


def maiac_annual_smoke_days(ds: xr.Dataset, years=None) -> xr.DataArray:
    """MAIAC smoke days summed to calendar years."""
    sd = maiac_smoke_days(ds)
    ann = sd.groupby(ds["time"].dt.year).sum("time")
    if years is not None:
        ann = ann.sel(year=list(years))
    return ann


# --------------------------------------------------------------------------
# Common grid
# --------------------------------------------------------------------------

def _hms_to_maiac_index(hms: xr.Dataset, maiac: xr.Dataset):
    """Map every HMS 0.1 deg cell centre to the 25 km Albers cell containing it.

    Binning by cell centre, not area-weighted resampling. This is the same
    choice the MAIAC pipeline makes going from 1 km to 25 km, and for the same
    reason: at a ~2.5:1 linear downsample every source cell is counted exactly
    once, with no resampling kernel to explain, and the error is confined to
    cell edges. Cells are area-weighted within the target cell by cos(lat),
    because a 0.1 deg box shrinks by ~30 % from the Mexican to the Canadian
    border and an unweighted mean would over-count the north.

    Returns (flat_index, weight, valid) over the flattened HMS grid.
    """
    lon2d, lat2d = np.meshgrid(hms["lon"].values, hms["lat"].values)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xx, yy = tr.transform(lon2d.ravel(), lat2d.ravel())

    x = maiac["x"].values
    y = maiac["y"].values
    dx = float(np.abs(np.diff(x)[0]))
    dy = float(np.abs(np.diff(y)[0]))
    # y descends, so its edges run the other way.
    ix = np.floor((xx - (x[0] - dx / 2)) / dx).astype(np.int64)
    iy = np.floor(((y[0] + dy / 2) - yy) / dy).astype(np.int64)

    inside = (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
    flat = np.where(inside, iy * x.size + ix, 0)
    weight = np.cos(np.deg2rad(lat2d.ravel()))
    return flat, weight, inside


def build_common_grid(cls: str = "heavy", years=None) -> xr.Dataset:
    """HMS annual smoke days and MAIAC annual smoke days on one 25 km grid.

    The returned dataset carries both fields plus `conus_fraction`, the share of
    each 25 km cell that is CONUS land according to the HMS mask. Everything
    downstream masks on `conus_fraction >= MIN_CONUS_FRACTION`.
    """
    years = list(range(FIRST_YEAR, LAST_FULL_YEAR + 1)) if years is None else list(years)
    hms = load_hms_grid(cls)
    maiac = load_maiac()

    flat, weight, inside = _hms_to_maiac_index(hms, maiac)
    ny, nx = maiac["y"].size, maiac["x"].size
    ncell = ny * nx

    in_conus = hms["in_conus"].values.ravel()
    use = inside & in_conus
    # Denominators: total cos-weight landing in each target cell, and the CONUS
    # share of it. A cell straddling the coast or the border gets a fraction
    # below 1 and is dropped by the mask above.
    w_all = np.bincount(flat[inside], weights=weight[inside], minlength=ncell)
    w_conus = np.bincount(flat[use], weights=weight[use], minlength=ncell)
    with np.errstate(invalid="ignore", divide="ignore"):
        conus_fraction = np.where(w_all > 0, w_conus / w_all, 0.0)

    stack = []
    for yr in years:
        v = hms[f"{cls}_smoke_days"].sel(year=yr).values.ravel().astype(float)
        num = np.bincount(flat[use], weights=(v * weight)[use], minlength=ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            stack.append(np.where(w_conus > 0, num / w_conus, np.nan))
    hms_days = np.array(stack).reshape(len(years), ny, nx)

    maiac_days = maiac_annual_smoke_days(maiac, years)

    out = xr.Dataset(
        {
            "hms_smoke_days": (("year", "y", "x"), hms_days),
            "maiac_smoke_days": (("year", "y", "x"), maiac_days.values),
            "conus_fraction": (("y", "x"), conus_fraction.reshape(ny, nx)),
        },
        coords={"year": years, "y": maiac["y"].values, "x": maiac["x"].values},
    )
    out.attrs["hms_density_class"] = cls
    out.attrs["regrid"] = ("HMS 0.1 deg cell centres binned to the 25 km EPSG:5070 "
                           "MAIAC grid, cos(lat)-weighted, CONUS cells only")
    out.attrs["min_conus_fraction"] = MIN_CONUS_FRACTION
    hms.close()
    maiac.close()
    return out


def conus_mask(grid: xr.Dataset) -> xr.DataArray:
    return grid["conus_fraction"] >= MIN_CONUS_FRACTION


def cell_edges(grid: xr.Dataset):
    """Cell edges for pcolormesh. `y` descends, so its edges descend too."""
    x = grid["x"].values
    y = grid["y"].values
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    xe = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])
    ye = np.concatenate([y + dy / 2, [y[-1] - dy / 2]])
    return xe, ye


def lonlat_of(grid: xr.Dataset):
    """Lon/lat of every cell centre, for regional masks and map annotation."""
    X, Y = np.meshgrid(grid["x"].values, grid["y"].values)
    tr = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    return tr.transform(X, Y)


# Longitude splits used for the regional breakdown. -105 is roughly the
# Continental Divide and -90 the Mississippi; both are round numbers chosen
# before any statistic was computed, not tuned to the answer.
REGIONS = {
    "West": lambda lon: lon < -105,
    "Central": lambda lon: (lon >= -105) & (lon < -90),
    "East": lambda lon: lon >= -90,
}


# --------------------------------------------------------------------------
# CONUS-aggregate monthly series
# --------------------------------------------------------------------------

def maiac_monthly_conus(ds: xr.Dataset | None = None) -> pd.DataFrame:
    """CONUS-aggregate MAIAC smoke statistics, one row per month.

    Ratio of sums, never a mean of per-cell ratios: the weights are summed over
    the domain first and divided afterwards, so a cell MODIS barely saw cannot
    shout as loudly as a well-observed one (../../modis/README.md, caveat 2).
    """
    close = ds is None
    ds = load_maiac() if ds is None else ds
    smoke = ds["smoke_pixel_day_weight"].sum(("y", "x"))
    valid = ds["valid_pixel_day_weight"].sum(("y", "x"))
    aod_sum = (ds["smoke_aod_index"] * ds["valid_pixel_day_weight"]).fillna(0.0).sum(("y", "x"))
    t = pd.DatetimeIndex(ds["time"].values)
    df = pd.DataFrame({
        "year": t.year, "month": t.month,
        "smoke_pixel_days": smoke.values,
        "valid_pixel_days": valid.values,
        "smoke_aod_sum": aod_sum.values,
    })
    df["smoke_fraction"] = df["smoke_pixel_days"] / df["valid_pixel_days"]
    df["smoke_aod_index"] = df["smoke_aod_sum"] / df["valid_pixel_days"]
    df["days_in_month"] = [calendar.monthrange(y, m)[1]
                           for y, m in zip(df["year"], df["month"])]
    df["maiac_smoke_days"] = df["smoke_fraction"] * df["days_in_month"]
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    if close:
        ds.close()
    return df


def hms_monthly_conus() -> pd.DataFrame:
    """CONUS-total HMS smoke-area-days by month, for each density class."""
    d = load_hms_daily()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    g = d.groupby(["year", "month"], as_index=False).agg(
        hms_all_km2_days=("all_km2", "sum"),
        hms_light_km2_days=("light_km2", "sum"),
        hms_heavy_km2_days=("heavy_km2", "sum"),
        n_days=("all_km2", "size"),
    )
    g["date"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    return g


def merged_monthly() -> pd.DataFrame:
    """The joined monthly table both time-series figures are built from."""
    m = maiac_monthly_conus()
    h = hms_monthly_conus()
    df = m.merge(h, on=["year", "month", "date"], how="inner")
    df = df[(df["year"] >= FIRST_YEAR) & (df["date"] <= MONTHLY_END)]
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def correlations(a, b) -> dict:
    """Pearson, Spearman and log-space Pearson, with n, on the common finite set.

    All three are reported everywhere rather than one being chosen per figure.
    Smoke series are heavy-tailed and near-zero for much of the year, so Pearson
    alone is dominated by a handful of fire months and Spearman alone throws
    away the size of them; log-Pearson sits between and needs an offset, which
    is set from the data rather than a round number.
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": int(ok.sum())}
    if out["n"] < 3:
        return {**out, "pearson_r": np.nan, "spearman_r": np.nan, "log_pearson_r": np.nan}
    r, p = stats.pearsonr(a, b)
    rho, prho = stats.spearmanr(a, b)
    pos_a = a[a > 0]
    pos_b = b[b > 0]
    off_a = pos_a.min() / 2 if pos_a.size else 1.0
    off_b = pos_b.min() / 2 if pos_b.size else 1.0
    lr, lp = stats.pearsonr(np.log10(a + off_a), np.log10(b + off_b))
    out.update({
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_r": float(rho), "spearman_p": float(prho),
        "log_pearson_r": float(lr), "log_pearson_p": float(lp),
    })
    return out


def theil_sen(x, y) -> dict:
    """Theil-Sen slope with a 95 % confidence interval, plus Mann-Kendall.

    Ordinary least squares is the wrong tool for a 14-point series with one or
    two enormous fire years in it; Theil-Sen is not moved by them.
    """
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope, intercept, lo, hi = stats.theilslopes(y, x, 0.95)
    tau, p = stats.kendalltau(x, y)
    return {"slope": float(slope), "intercept": float(intercept),
            "slope_lo": float(lo), "slope_hi": float(hi),
            "kendall_tau": float(tau), "kendall_p": float(p), "n": int(x.size)}
