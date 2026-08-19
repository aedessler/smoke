#!/usr/bin/env python3
"""Definitions specific to the MAIAC smoke-record paper.

The loaders, the regridding, the smoke-day conversion and the design tokens
live in common.py alongside this module and are re-exported here rather than
reimplemented. Everything defined *in* this module is new to this paper: the
analysis window, the analysis domain, and the per-cell trend machinery.

    from maiac_common import C          # the shared module, unchanged
    from maiac_common import domain, annual_cell_days, sen_slope
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

FIG_DIR = PAPER / "figures"
RESULT_DIR = PAPER / "results"

# --------------------------------------------------------------------------
# Analysis window
# --------------------------------------------------------------------------
#
# The archive runs 2000-02 to 2025-07, but the trend window is 2003-2024 and
# the reason is not arbitrary. `valid_pixel_day_weight` counts pixel-*days*
# with at least one good observation, and `smoke_pixel_day_weight` counts
# pixel-days with at least one smoke-model observation. Aqua joins Terra in
# July 2002, so from 2003 onward a pixel-day has two chances to be seen clear
# and two chances to be labelled smoke instead of one. That raises both
# accumulators, and it raises the smoke one more, because a brief plume that
# Terra missed can still be caught by Aqua. The ratio is therefore not
# homogeneous across mid-2002 and no trend should be estimated across it.
#
# 2025 is excluded from annual statistics because the archive stops in July,
# which truncates the record inside the August-September smoke peak.
FIRST_YEAR = 2003
LAST_YEAR = 2024
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)

ARCHIVE_START = "2000-02"
ARCHIVE_END = "2025-07"
TWO_SATELLITE_START = 2003

CELL_KM = 25.0
CELL_AREA_KM2 = CELL_KM ** 2

# Colours for this paper. The shared module's HMS/MAIAC pair is kept for the
# validation section; the sequential ramps below are used for the maps.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = C.INK, C.INK_2, C.MUTED, C.GRID, C.AXIS, C.SURFACE
SMOKE = C.MAIAC          # the MAIAC series colour, used for everything MODIS
HMS_C = C.HMS
WARM = "#b4451c"         # increases
COOL = "#2a6fa8"         # decreases
DPI = C.DPI

MONTH_ABBR = C.MONTH_ABBR
style_axes = C.style_axes


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  wrote figures/{name}")
    return out


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

def domain(ds: xr.Dataset | None = None) -> xr.DataArray:
    """Cells carrying at least one valid pixel-day in every year of the window.

    This is MODIS's own CONUS footprint -- the mask the MAIAC pipeline applied
    at 1 km, seen through the requirement that a cell be observed in all 22
    years so that a per-cell trend is defined on a fixed sample. No external
    boundary file is involved, which matters because Section 5 compares this
    grid with a product that has its own CONUS mask; keeping the two separate
    is what makes that comparison a check rather than a tautology.

    The 8 cells this drops relative to "observed at least once" are all coastal
    slivers.
    """
    close = ds is None
    ds = C.load_maiac() if ds is None else ds
    sub = ds.sel(time=slice(f"{FIRST_YEAR}-01", f"{LAST_YEAR}-12"))
    v = sub["valid_pixel_day_weight"]
    ann = v.groupby(v["time"].dt.year).sum("time")
    out = (ann.min("year") > 0).compute()
    out.name = "domain"
    if close:
        ds.close()
    return out


def annual_cell_days(ds: xr.Dataset | None = None) -> xr.DataArray:
    """Per-cell smoke days per calendar year, (year, y, x), over the window.

    Smoke days for a month are the clear-sky smoke frequency times the length
    of the month; the year is their sum. Weighting each month by its own length
    rather than by its observation count is deliberate: the quantity being
    estimated is "how many days of this year carried smoke over this cell", and
    a cloudy month is still a month.
    """
    close = ds is None
    ds = C.load_maiac() if ds is None else ds
    sub = ds.sel(time=slice(f"{FIRST_YEAR}-01", f"{LAST_YEAR}-12"))
    sd = C.maiac_smoke_days(sub)
    out = sd.groupby(sub["time"].dt.year).sum("time")
    out.name = "smoke_days"
    if close:
        ds.close()
    return out


def conus_monthly(ds: xr.Dataset | None = None, mask: np.ndarray | None = None):
    """Ratio of sums over the analysis domain, one row per month of the archive.

    Summing the weights over the domain before dividing is what keeps a cell
    MODIS barely saw from shouting as loudly as a well-observed one. The four
    accumulated columns are the only things summed; every ratio in this paper
    is formed from them afterwards, here or in run_analysis.py.
    """
    import pandas as pd
    close = ds is None
    ds = C.load_maiac() if ds is None else ds
    mask = domain(ds).values if mask is None else mask
    m = xr.DataArray(mask, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"))
    valid = ds["valid_pixel_day_weight"].where(m).sum(("y", "x"))
    smoke = ds["smoke_pixel_day_weight"].where(m).sum(("y", "x"))
    # smoke_aod_index is sum(C)/sum(B) per cell, so multiplying by B recovers
    # sum(C): the smoke-model AOD accumulated over the cell-month.
    aod_c = ((ds["smoke_aod_index"] * ds["valid_pixel_day_weight"])
             .where(m).fillna(0.0).sum(("y", "x")))
    aod_a = ((ds["mean_aod_055"] * ds["valid_pixel_day_weight"])
             .where(m).fillna(0.0).sum(("y", "x")))
    t = pd.DatetimeIndex(ds["time"].values)
    df = pd.DataFrame({
        "date": t, "year": t.year, "month": t.month,
        "valid_pixel_days": valid.values,
        "smoke_pixel_days": smoke.values,
        "sum_smoke_aod": aod_c.values,
        "sum_aod": aod_a.values,
        "days_in_month": t.days_in_month,
    })
    df["smoke_fraction"] = df["smoke_pixel_days"] / df["valid_pixel_days"]
    df["smoke_days"] = df["smoke_fraction"] * df["days_in_month"]
    df["smoke_aod_index"] = df["sum_smoke_aod"] / df["valid_pixel_days"]
    df["mean_aod"] = df["sum_aod"] / df["valid_pixel_days"]
    # AOD conditional on the smoke model being selected. Undefined, not zero,
    # in a month with no smoke anywhere -- which never happens at CONUS scale
    # but does happen cell by cell, and is why this field is never mapped.
    df["smoke_intensity"] = np.where(df["smoke_pixel_days"] > 0,
                                     df["sum_smoke_aod"] / df["smoke_pixel_days"],
                                     np.nan)
    if close:
        ds.close()
    return df


def annual_from_monthly(mon):
    """Calendar-year aggregation, ratios re-formed from the summed accumulators."""
    g = mon.groupby("year", as_index=False).agg(
        n_months=("month", "size"),
        valid_pixel_days=("valid_pixel_days", "sum"),
        smoke_pixel_days=("smoke_pixel_days", "sum"),
        sum_smoke_aod=("sum_smoke_aod", "sum"),
        sum_aod=("sum_aod", "sum"),
        smoke_days=("smoke_days", "sum"),
    )
    g["smoke_aod_index"] = g["sum_smoke_aod"] / g["valid_pixel_days"]
    g["mean_aod"] = g["sum_aod"] / g["valid_pixel_days"]
    g["smoke_intensity"] = g["sum_smoke_aod"] / g["smoke_pixel_days"]
    return g


def window(df):
    """Rows of a year-indexed frame inside the trend window."""
    return df[(df["year"] >= FIRST_YEAR) & (df["year"] <= LAST_YEAR)]


def masked_stack(ds: xr.Dataset | None = None):
    """(years, values[nyear, ncell], mask, lon, lat) -- the analysis array."""
    close = ds is None
    ds = C.load_maiac() if ds is None else ds
    mask = domain(ds).values
    ann = annual_cell_days(ds)
    years = ann["year"].values.astype(float)
    vals = ann.values.reshape(ann.shape[0], -1)[:, mask.ravel()]
    lon, lat = C.lonlat_of(ds)
    if close:
        ds.close()
    return years, vals, mask, lon.ravel()[mask.ravel()], lat.ravel()[mask.ravel()]


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def sen_slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Theil-Sen slope for every column of `y` at once.

    scipy.stats.theilslopes handles one series at a time and 13,064 calls is
    slow; the pairwise-slope median is two lines when the time axis is shared.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i, j = np.triu_indices(x.size, 1)
    return np.median((y[j] - y[i]) / (x[j] - x[i])[:, None], axis=0)


def kendall_stack(y: np.ndarray, x: np.ndarray):
    """Kendall tau and its two-sided p-value for every column of `y`."""
    from scipy import stats
    n = y.shape[1]
    tau = np.empty(n)
    p = np.empty(n)
    for k in range(n):
        tau[k], p[k] = stats.kendalltau(x, y[:, k])
    return tau, p


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: which of `p` are rejected controlling FDR at q.

    Thirteen thousand grid cells means thirteen thousand hypothesis tests, and
    at p < 0.05 some 650 cells would be flagged by chance alone. BH is valid
    under the positive dependence that spatial autocorrelation produces, so it
    is the right multiplicity correction here -- but it is a correction for
    multiplicity only. It asks whether an individual cell can be certified,
    which on 22 noisy years is a demanding question; `field_test` below asks the
    different and, for a trend map, more useful one.
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = p.size
    thresh = q * np.arange(1, n + 1) / n
    passed = np.isfinite(ranked) & (ranked <= thresh)
    out = np.zeros(n, dtype=bool)
    if passed.any():
        cut = np.max(np.where(passed)[0])
        out[order[: cut + 1]] = True
    return out


def spearman_with_time(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank correlation with time for every column of `y`, vectorised."""
    from scipy.stats import rankdata
    r = rankdata(y, axis=0)
    rz = (r - r.mean(0)) / np.where(r.std(0) > 0, r.std(0), np.nan)
    xz = (x - x.mean()) / x.std()
    return (xz @ rz) / x.size


def field_test(y: np.ndarray, x: np.ndarray, n_perm: int = 2000,
               alpha: float = 0.05, seed: int = 0) -> dict:
    """Permutation test for a trend *map*, in the sense of Livezey and Chen.

    Shuffling the year labels destroys any trend while leaving the spatial
    correlation of the field exactly intact, so the null distribution built
    this way already contains whatever clustering the domain has. That is what
    a per-cell p-value cannot do: neighbouring 25 km cells see the same plumes
    and are nowhere near independent, so counting cells below p < 0.05 against
    a nominal 5 % is not a test of anything.

    Two questions are answered with one set of permutations:

      * per cell -- an exact permutation p-value for the rank correlation with
        time, which needs no distributional assumption and handles the heavy
        ties that a cell with many smoke-free years produces;
      * per field -- whether the *number* of cells reaching local significance,
        and the *fraction* of cells trending upward, are larger than the
        shuffled field produces.

    Rank correlation is used rather than Kendall's tau only because permuting
    it is a matrix product: the data ranks are fixed under a permutation of the
    time axis, so all `n_perm` maps are one (n_perm x nt) @ (nt x ncell) call.
    """
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    nt = x.size

    r = rankdata(y, axis=0)
    sd = r.std(0)
    live = sd > 0
    rz = np.zeros_like(r)
    rz[:, live] = (r[:, live] - r[:, live].mean(0)) / sd[live]

    xz = (x - x.mean()) / x.std()
    obs = (xz @ rz) / nt

    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(xz) for _ in range(n_perm)])
    null = (perms @ rz) / nt                       # (n_perm, ncell)

    p_cell = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (n_perm + 1)
    crit = np.quantile(np.abs(null), 1 - alpha, axis=0)
    obs_frac_sig = float(np.mean(np.abs(obs[live]) >= crit[live]))
    null_frac_sig = (np.abs(null[:, live]) >= crit[live]).mean(1)
    obs_frac_up = float(np.mean(obs[live] > 0))
    null_frac_up = (null[:, live] > 0).mean(1)

    return {
        "rho": obs,
        "p_cell": np.where(live, p_cell, np.nan),
        "live": live,
        "n_cells": int(live.sum()),
        "n_constant_cells": int((~live).sum()),
        "n_perm": n_perm,
        "alpha": alpha,
        "frac_locally_significant": obs_frac_sig,
        "frac_locally_significant_null_p95": float(np.quantile(null_frac_sig, 0.95)),
        "frac_locally_significant_p": float(
            (1 + (null_frac_sig >= obs_frac_sig).sum()) / (n_perm + 1)),
        "frac_upward": obs_frac_up,
        "frac_upward_null_p95": float(np.quantile(null_frac_up, 0.95)),
        "frac_upward_p": float((1 + (null_frac_up >= obs_frac_up).sum()) / (n_perm + 1)),
    }


def log_trend(y: np.ndarray, x: np.ndarray) -> dict:
    """Theil-Sen on log(y): a proportional rate, reported as percent per year.

    Smoke is bounded below by zero and varies by more than an order of
    magnitude between quiet and severe years, so an additive slope in days per
    year is a poor summary of it on its own. The multiplicative rate is
    reported alongside, never instead.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0)
    d = C.theil_sen(np.asarray(x, dtype=float)[ok], np.log(y[ok]))
    span = float(np.nanmax(x) - np.nanmin(x))
    return {
        "percent_per_year": 100.0 * d["slope"],
        "percent_per_year_lo": 100.0 * d["slope_lo"],
        "percent_per_year_hi": 100.0 * d["slope_hi"],
        "factor_over_window": float(np.exp(d["slope"] * span)),
        "kendall_tau": d["kendall_tau"],
        "kendall_p": d["kendall_p"],
        "n": d["n"],
    }
