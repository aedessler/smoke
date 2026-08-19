#!/usr/bin/env python3
"""Change in MAIAC smoke frequency between two periods, as one CONUS map.

The MAIAC counterpart to ../noaa/plot_period_change.py. Defaults to
2021-2025 minus 2006-2010, five-year windows at each end of the overlap with HMS.

WHAT IS DIFFERENCED. Per-cell smoke frequency: the share of that cell's valid
pixel-days on which MAIAC selected a smoke aerosol model, formed as a ratio of
sums over every month in the window (modis/README.md, caveat 2). The difference is
in PERCENTAGE POINTS of that share -- not in days, which MAIAC cannot give: a cell
observed on 40 % of days and smoky on half of those has no defensible conversion
to "days per year" without assuming the unobserved days behaved like the observed
ones, and cloud and snow are exactly when they do not.

WHY THIS ONE IS NOT DRAWN IN GRAY. The HMS version of this map uses a sequential
ramp because every CONUS cell increased there. MAIAC disagrees: about 6 % of cells
fall. A sequential ramp would paint those the same near-white as "no change", so
this map defaults to a diverging ramp, symmetric about a neutral zero, and the
zero contour is drawn. --cmap gray forces the HMS look for side-by-side viewing,
at the cost of hiding the sign.

2025 IS PARTIAL (Jan-Jul). It is included by default because the window is a
ratio of sums, so the seven months contribute seven months' worth to both halves
of the ratio -- but they tilt the late window toward the low-smoke half of the
year. The figure says so, and --skip-partial drops any incomplete year to check.

Run from the modis/ folder (paths resolve relative to it either way):

    python3 maiac/plot_period_change.py
    python3 maiac/plot_period_change.py --late 2021-2024 --cmap gray
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

mpl.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maiac_pipeline import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the modis/ folder
DEFAULT_NC = ROOT / "data" / "maiac_smoke_25km_monthly.nc"
FIG_DIR = ROOT / "figures"

# Tokens duplicated from the noaa figures by design -- the top-level README makes
# it a rule that the two products share no code.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"

SEQ_GRAY = [
    "#e4e3df", "#d5d4cf", "#c5c4be", "#b4b3ad", "#a3a29c", "#92918b",
    "#82817b", "#71706b", "#605f5a", "#50504b", "#41403c", "#32312e", "#232220",
]
# Two hues meeting at a near-neutral middle: never a hue at zero, or "no change"
# reads as a value of its own.
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#eceae6", "#f6bd9a", "#e88a4e", "#c04d16"]

CMAPS = {
    "diverging": LinearSegmentedColormap.from_list("div_blue_orange", DIVERGING),
    "gray": LinearSegmentedColormap.from_list("seq_gray", SEQ_GRAY),
}
for _cm in CMAPS.values():
    _cm.set_bad(SURFACE)

DPI = 150
MIN_WEIGHT = 1000.0  # valid pixel-days per cell per window; below this a cell is noise
CONTOUR_SMOOTH = 1.2  # in 25 km cells
MIN_CONTOUR_SPAN = 150_000.0  # metres; shorter closed rings are noise, not structure

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
})


def parse_period(text: str) -> tuple[int, int]:
    try:
        y0, y1 = (int(p) for p in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-YYYY, got {text!r}") from None
    if y1 < y0:
        raise argparse.ArgumentTypeError(f"{text}: end year precedes start year")
    return y0, y1


def window(ds, period, skip_partial=False):
    """Smoke frequency (%) per cell over the window, plus its valid-day weight.

    Ratio of sums: total smoke pixel-days over total valid pixel-days across every
    month in the window. Never a mean of the monthly per-cell ratios.
    """
    y0, y1 = period
    year = ds["time"].dt.year
    sel = (year >= y0) & (year <= y1)
    if skip_partial:
        months = ds["time"].dt.month.groupby(year).count()
        incomplete = [int(y) for y in months["year"].values if int(months.sel(year=y)) < 12]
        if incomplete:
            sel = sel & ~year.isin(incomplete)
    valid = ds["valid_pixel_day_weight"].where(sel).sum("time").values
    smoke = ds["smoke_pixel_day_weight"].where(sel).sum("time").values
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = 100.0 * smoke / np.where(valid > 0, valid, np.nan)
    n_months = int(sel.sum())
    return freq, valid, n_months


def states():
    """CONUS state outlines in the grid CRS, or None. Cached by plot_month.py."""
    try:
        import geopandas as gpd

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".plotcache", "states.geojson")
        if not os.path.exists(path):
            print("  (state outlines not cached; run maiac/plot_month.py once)")
            return None
        return gpd.read_file(path).to_crs("EPSG:5070")
    except Exception as exc:  # outlines are decoration, not the point
        print(f"  (state outlines unavailable: {exc})")
        return None


def draw_contours(ax, field, extent, levels):
    """White contours with a dark halo, labeled inline.

    The halo is what makes one contour color work on both ramps. Plain white
    disappears against the near-white middle of the diverging ramp -- which is
    exactly where the zero line lives, the one contour that must be readable --
    and against the pale end of the gray one. A thin dark stroke under a white
    core reads on every step of either.
    """
    filled = np.where(np.isfinite(field), field, np.nan)
    smooth = gaussian_filter(np.nan_to_num(filled, nan=0.0), CONTOUR_SMOOTH)
    weight = gaussian_filter(np.isfinite(filled).astype(float), CONTOUR_SMOOTH)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(weight > 0.35, smooth / weight, np.nan)
    smooth = np.where(np.isnan(filled), np.nan, smooth)

    x = np.linspace(extent[0], extent[1], field.shape[1])
    y = np.linspace(extent[3], extent[2], field.shape[0])
    halo = [pe.withStroke(linewidth=3.0, foreground="#3a3a38", alpha=0.5)]

    cs = ax.contour(x, y, np.ma.masked_invalid(smooth), levels=levels,
                    colors="white", linewidths=1.4, zorder=4)
    kept = []
    for path in cs.get_paths():
        pieces = [
            MplPath(v) for v in path.to_polygons(closed_only=False)
            if len(v) > 1 and np.ptp(v, axis=0).max() >= MIN_CONTOUR_SPAN
        ]
        kept.append(MplPath.make_compound_path(*pieces) if pieces else MplPath([]))
    cs.set_paths(kept)
    cs.set_path_effects(halo)

    labels = ax.clabel(cs, fmt="%g", fontsize=8, colors="white", inline=True,
                       inline_spacing=4)
    for text in labels:
        text.set_path_effects(halo)
    return cs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("netcdf", nargs="?", default=str(DEFAULT_NC))
    ap.add_argument("--early", default="2006-2010", type=parse_period)
    ap.add_argument("--late", default="2021-2025", type=parse_period)
    ap.add_argument("--cmap", default="diverging", choices=sorted(CMAPS),
                    help="diverging (default, keeps the sign) or gray (HMS look)")
    ap.add_argument("--min-weight", type=float, default=MIN_WEIGHT,
                    help=f"drop cells with fewer valid pixel-days per window "
                         f"(default: {MIN_WEIGHT:g})")
    ap.add_argument("--skip-partial", action="store_true",
                    help="exclude any year with fewer than 12 months")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args(argv)

    src = Path(args.netcdf)
    if not src.exists():
        print(f"ERROR: {src} not found.", file=sys.stderr)
        return 1

    ds = xr.open_dataset(src)
    covered = {int(y): int(n) for y, n in
               zip(*[v.values for v in (ds["time"].dt.year.groupby(ds["time"].dt.year).first(),
                                        ds["time"].dt.month.groupby(ds["time"].dt.year).count())])}
    f_early, w_early, n_early = window(ds, args.early, args.skip_partial)
    f_late, w_late, n_late = window(ds, args.late, args.skip_partial)

    # A cell needs real observation in BOTH windows to have a difference. Without
    # this a handful of cells with single-digit pixel-days produce frequencies of
    # 50 or 100 % and set the color scale on their own.
    thin = (w_early < args.min_weight) | (w_late < args.min_weight)
    diff = np.where(thin, np.nan, f_late - f_early)
    good = np.isfinite(diff)
    v = diff[good]
    dropped = int((thin & ((w_early > 0) | (w_late > 0))).sum())
    print(f"  {int(good.sum()):,} cells kept, {dropped:,} dropped as thin "
          f"(< {args.min_weight:g} valid pixel-days in a window)")
    print(f"  change: mean {v.mean():+.2f} pp, range {v.min():+.2f} to {v.max():+.2f}, "
          f"{100 * (v < 0).mean():.1f}% of cells fell")

    two_signed = v.min() < 0 < v.max()
    lim = float(np.ceil(np.percentile(np.abs(v), 99)))
    if args.cmap == "diverging" and two_signed:
        vmin, vmax, extend = -lim, lim, "both"
        levels = [x for x in (0, 2.5, 5, 7.5, 10) if vmin < x < vmax or x == 0]
    else:
        vmin, vmax, extend = 0.0, lim, "max"
        levels = [x for x in (2.5, 5, 7.5, 10) if x < vmax]
        if two_signed:
            print("  NOTE: field has both signs but a sequential ramp was requested; "
                  "decreases will render as 'no change'.")

    extent = [config.TARGET_XMIN, config.TARGET_XMAX,
              config.TARGET_YMIN, config.TARGET_YMAX]
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=DPI)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=0.20)
    im = ax.imshow(np.ma.masked_invalid(diff), extent=extent, origin="upper",
                   cmap=CMAPS[args.cmap], vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=1)
    st = states()
    if st is not None:
        st.boundary.plot(ax=ax, linewidth=0.4,
                         edgecolor="white" if args.cmap == "gray" else INK_2,
                         alpha=0.55, zorder=2)
    draw_contours(ax, diff, extent, levels)

    # The grid rectangle runs well past the data on every side -- north into
    # Canada, south into the Gulf -- so framing on the rectangle wastes most of
    # the panel. Frame on the cells that actually carry data instead.
    rows, cols = np.where(good)
    pad = 3 * float(abs(ds["x"].values[1] - ds["x"].values[0]))
    xs, ys = ds["x"].values, ds["y"].values
    ax.set_xlim(xs[cols.min()] - pad, xs[cols.max()] + pad)
    ax.set_ylim(ys[rows.max()] - pad, ys[rows.min()] + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    (y0e, y1e), (y0l, y1l) = args.early, args.late
    fig.suptitle("Change in smoke frequency over the contiguous United States",
                 fontsize=14, color=INK, x=0.5, y=0.965)
    fig.text(0.5, 0.915, f"average {y0l}–{y1l} minus average {y0e}–{y1e}",
             ha="center", fontsize=10.5, color=INK_2)
    fig.text(0.5, 0.876,
             f"{100 * (v > 0).mean():.0f}% of CONUS cells rose, "
             f"{100 * (v < 0).mean():.0f}% fell; range {v.min():+.1f} to {v.max():+.1f} "
             f"points, mean {v.mean():+.1f}",
             ha="center", fontsize=9.5, color=INK_2)

    partial = [y for y in range(y0l, y1l + 1) if covered.get(y, 0) < 12]
    if partial and not args.skip_partial:
        fig.text(0.5, 0.845,
                 f"{', '.join(str(y) for y in partial)} partial "
                 f"({covered[partial[0]]} months); included, so the late window leans "
                 "toward the low-smoke half of the year",
                 ha="center", fontsize=8.5, color=MUTED)

    cax = fig.add_axes([0.30, 0.115, 0.40, 0.026])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(f"change in % of valid pixel-days under smoke "
                 f"({y0l}–{y1l} − {y0e}–{y1e})", color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=0.8, color=MUTED, labelcolor=MUTED)

    out = Path(args.out) if args.out else FIG_DIR / "maiac_smoke_frequency_change.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
