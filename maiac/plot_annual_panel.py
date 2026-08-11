#!/usr/bin/env python3
"""Year-by-year facet of MAIAC smoke frequency over CONUS.

The MAIAC counterpart to the HMS ../noaa/ smoke-days panel: one small map per
year, all on one shared color scale so the years are directly comparable. Covers
the whole record, 2000-02 through 2025-07.

WHAT EACH CELL SHOWS. The share of that cell's valid pixel-days in the year on
which MAIAC selected a smoke aerosol model, as a ratio of sums over the year's
months. Not "days per year": MAIAC sees a cell only when cloud, snow and orbit
allow, and converting a rate into days would mean assuming the unobserved days
behaved like the observed ones -- which is wrong in exactly the season and region
where the gap is biggest.

TWO DELIBERATE DIFFERENCES FROM THE HMS PANEL.

Zeros are drawn, not masked. The HMS panel hides zero-day cells because there the
lightest ramp step over the whole country would drown the low counts. Here a zero
means "MAIAC watched this cell all year and never saw smoke", which is a real
result and common in the quiet years; masking it would turn 2001 and 2010 into
maps that look like missing data instead of clean air.

Thinly-observed cells ARE masked (--min-weight). A cell with a few hundred valid
pixel-days in a year produces a frequency as confidently as one with 90,000 --
modis/README.md, caveat 3.

Run from the modis/ folder (paths resolve relative to it either way):

    python3 maiac/plot_annual_panel.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maiac_pipeline import config  # noqa: E402
from plot_period_change import (  # noqa: E402  -- same folder, same product
    CMAPS,
    DPI,
    FIG_DIR,
    INK,
    INK_2,
    MUTED,
    ROOT,
    states,
)

DEFAULT_NC = ROOT / "data" / "maiac_smoke_25km_monthly.nc"
MIN_WEIGHT = 200.0  # valid pixel-days per cell per YEAR
NCOL = 4
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def annual_frequency(ds, min_weight: float):
    """(years, freq[y,row,col] in %, span label per year)."""
    year = ds["time"].dt.year
    valid = ds["valid_pixel_day_weight"].groupby(year).sum("time")
    smoke = ds["smoke_pixel_day_weight"].groupby(year).sum("time")
    years = [int(y) for y in valid["year"].values]

    v, s = valid.values, smoke.values
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = 100.0 * s / np.where(v > 0, v, np.nan)
    freq = np.where(v >= min_weight, freq, np.nan)

    months = ds["time"].dt.month.groupby(year)
    spans = {}
    for y, g in months:
        m = sorted(int(x) for x in g.values)
        spans[int(y)] = None if len(m) == 12 else f"{MONTHS[m[0] - 1]}–{MONTHS[m[-1] - 1]}"
    return years, freq, spans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("netcdf", nargs="?", default=str(DEFAULT_NC))
    ap.add_argument("--cmap", default="gray", choices=sorted(CMAPS))
    ap.add_argument("--min-weight", type=float, default=MIN_WEIGHT,
                    help=f"drop cells with fewer valid pixel-days in the year "
                         f"(default: {MIN_WEIGHT:g})")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args(argv)

    src = Path(args.netcdf)
    if not src.exists():
        print(f"ERROR: {src} not found.", file=sys.stderr)
        return 1

    ds = xr.open_dataset(src)
    years, freq, spans = annual_frequency(ds, args.min_weight)

    # One scale for every panel, or the years are not comparable. vmax from a high
    # percentile of the nonzero cells so a single saturated year cannot flatten
    # the other twenty-five.
    finite = freq[np.isfinite(freq)]
    vmax = float(np.ceil(np.percentile(finite[finite > 0], 99)))
    print(f"  shared color scale 0 -> {vmax:.0f}% "
          f"(99th pct of nonzero cells; max {finite.max():.1f}%)")

    extent = [config.TARGET_XMIN, config.TARGET_XMAX,
              config.TARGET_YMIN, config.TARGET_YMAX]
    xs, ys = ds["x"].values, ds["y"].values
    rows, cols = np.where(np.isfinite(freq).any(axis=0))
    pad = 3 * float(abs(xs[1] - xs[0]))
    xlim = (xs[cols.min()] - pad, xs[cols.max()] + pad)
    ylim = (ys[rows.max()] - pad, ys[rows.min()] + pad)

    st = states()
    nrow = int(np.ceil(len(years) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(13, 2.3 * nrow), dpi=DPI,
                             squeeze=False)
    for ax, year, field in zip(axes.ravel(), years, freq):
        im = ax.imshow(np.ma.masked_invalid(field), extent=extent, origin="upper",
                       cmap=CMAPS[args.cmap], vmin=0, vmax=vmax,
                       interpolation="nearest", zorder=1)
        if st is not None:
            st.boundary.plot(ax=ax, linewidth=0.3,
                             edgecolor="white" if args.cmap == "gray" else INK_2,
                             alpha=0.6, zorder=2)
        label = str(year) if spans[year] is None else f"{year}  ({spans[year]})"
        ax.set_title(label, fontsize=11, color=INK, pad=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    for ax in axes.ravel()[len(years):]:
        ax.set_visible(False)

    top = 1 - 0.55 / fig.get_figheight()
    fig.suptitle("Smoke frequency by year, contiguous United States",
                 fontsize=14, color=INK, x=0.5, y=top + 0.020)
    fig.text(0.5, top - 0.004,
             "MAIAC MCD19A2.061 — share of valid pixel-days retrieved with a smoke "
             "aerosol model, 25 km grid",
             ha="center", fontsize=9.5, color=INK_2)

    fig.subplots_adjust(left=0.02, right=0.98, top=top - 0.020,
                        bottom=0.078, wspace=0.03, hspace=0.12)
    cax = fig.add_axes([0.30, 0.047, 0.40, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("% of valid pixel-days under smoke", color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=0.8, color=MUTED, labelcolor=MUTED)
    fig.text(0.5, 0.009,
             "Bracketed years are partial. Before mid-2002 the record is Terra-only; "
             "Aqua roughly doubles observations per pixel-day from then on.",
             ha="center", va="bottom", fontsize=8.5, color=MUTED)

    out = Path(args.out) if args.out else FIG_DIR / "maiac_smoke_frequency_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
