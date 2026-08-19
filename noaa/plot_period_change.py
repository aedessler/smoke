#!/usr/bin/env python3
"""Average smoke days per year, early period vs recent period, and their difference.

Defaults to 2006-2009 vs 2021-2024 -- the first four years of the record against
the four most recent complete ones. Writes two figures:

  <cls>_smoke_days_period_change.png      the two periods side by side
  <cls>_smoke_days_period_difference.png  late minus early, one map

Both panels of the first share one color scale, so they are directly comparable;
the CONUS mean under each title carries the magnitude the eye cannot read off a
choropleth.

The difference map is drawn with the SEQUENTIAL ramp anchored at zero, not a
diverging one, because over these defaults the change is positive in every CONUS
cell (min about +6 days/yr). A diverging scale centered on zero would spend half
its range on empty space and imply a sign change the data does not contain. If a
period pair does produce both signs, the script says so and switches to a
symmetric diverging scale.

Uses the gridded day counts rather than the area series on purpose. The README's
central caveat is that analysts shifted to fewer, much larger polygons after ~2020,
which inflates the recent end of the AREA series; day counts are far less sensitive
to that, since one giant polygon and twenty small ones covering the same cell both
score that cell one day. Less sensitive is not immune -- a single saturating polygon
still marks every cell it spans -- and the early period predates GOES-16/17, so the
two windows straddle a genuine change in observing practice. The figure says so.

Needs data/processed/all_smoke_days.nc (run grid_all_smoke.py).

Run: python3 plot_period_change.py [--early 2006-2009] [--late 2021-2024]
                                   [--cmap gray|blue]
"""

from __future__ import annotations

import argparse
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

from plot_smoke import (
    DPI,
    FIG_DIR,
    IN_DIR,
    INK,
    INK_2,
    MUTED,
    SPECS,
    SURFACE,
    THEMES,
    draw_map,
)

PROJ = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))

# Diverging ramp for the two-signed case only: the house categorical blue and orange
# as the two poles, meeting at a near-neutral midpoint. Two hues and a neutral middle,
# never a hue at zero -- otherwise "no change" reads as a value of its own.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_orange",
    ["#184f95", "#3987e5", "#9ec5f4", "#eceae6", "#f6bd9a", "#e88a4e", "#c04d16"],
)
DIVERGING.set_bad(SURFACE)


def parse_period(text: str) -> tuple[int, int]:
    """'2006-2009' -> (2006, 2009)."""
    try:
        y0, y1 = (int(p) for p in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-YYYY, got {text!r}") from None
    if y1 < y0:
        raise argparse.ArgumentTypeError(f"{text}: end year precedes start year")
    return y0, y1


def period_mean(ds, var, period):
    """Mean days per year over the period, masked outside CONUS and at true zeros.

    Zeros are masked for the same reason the per-year maps mask them: the lightest
    ramp step is still a visible tint, so drawing genuine zeros washes the whole
    country and the low counts stop being readable.
    """
    y0, y1 = period
    have = [int(y) for y in ds["year"].values]
    missing = [y for y in range(y0, y1 + 1) if y not in have]
    if missing:
        raise SystemExit(
            f"ERROR: {var} has no data for {', '.join(str(y) for y in missing)}; "
            f"the file covers {min(have)}-{max(have)}."
        )
    field = ds[var].sel(year=slice(y0, y1)).mean("year").values.astype(float)
    inside = ds["in_conus"].values.astype(bool)
    return np.ma.masked_where(~inside | (field == 0), field), field[inside].mean()


def period_diff(ds, var, early, late):
    """Late-minus-early mean days per year, masked only outside CONUS.

    Zeros are NOT masked here, unlike the period panels: on a difference map zero
    is a real, interesting value -- "this cell did not change" -- and hiding it
    would punch holes wherever the answer is 'no change'.
    """
    y0e, y1e = early
    y0l, y1l = late
    e = ds[var].sel(year=slice(y0e, y1e)).mean("year").values.astype(float)
    l = ds[var].sel(year=slice(y0l, y1l)).mean("year").values.astype(float)
    inside = ds["in_conus"].values.astype(bool)
    return np.ma.masked_where(~inside, l - e)


CONTOUR_LEVELS = (50, 100, 125, 150)
CONTOUR_SMOOTH = 2.5  # grid cells (0.1 deg each), i.e. a ~0.25 deg gaussian
MIN_CONTOUR_SPAN = 2.0  # degrees; shorter closed loops are dropped as noise


def drop_tiny_contours(cs, min_span=MIN_CONTOUR_SPAN):
    """Remove contour rings too small to mean anything.

    A local wobble of a few cells closes into a ring a few tenths of a degree
    across. At figure scale those read as stray punctuation, not as structure, so
    anything whose bounding box is smaller than `min_span` degrees is dropped.
    """
    kept_paths = []
    for path in cs.get_paths():
        pieces = [
            Path(v) for v in path.to_polygons(closed_only=False)
            if len(v) > 1 and np.ptp(v, axis=0).max() >= min_span
        ]
        kept_paths.append(Path.make_compound_path(*pieces) if pieces else Path([]))
    cs.set_paths(kept_paths)
    return cs


def draw_contours(ax, field, lon, lat, levels=CONTOUR_LEVELS):
    """White contours over the shading, labeled inline.

    The 0.1 deg field is noisy at the scale of a contour line -- raw isolines come
    out as hairballs -- so the lines are drawn on a lightly smoothed copy. The
    shading underneath stays unsmoothed: the smoothing is a legibility device for
    the isolines, not a change to what the map reports.

    Drawn heavier than the state borders (which are also near-white in the gray
    theme) so the two read as different families of line rather than one mesh.
    """
    filled = field.filled(np.nan)
    smooth = gaussian_filter(np.nan_to_num(filled, nan=0.0), CONTOUR_SMOOTH)
    weight = gaussian_filter((~np.isnan(filled)).astype(float), CONTOUR_SMOOTH)
    # Normalize by the smoothed mask so cells near the coast are not dragged toward
    # zero by the ocean sitting in the kernel.
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(weight > 0.35, smooth / weight, np.nan)
    smooth = np.ma.masked_invalid(np.where(np.isnan(filled), np.nan, smooth))

    cs = ax.contour(
        lon, lat, smooth, levels=list(levels), colors="white", linewidths=1.1,
        transform=ccrs.PlateCarree(), zorder=3,
    )
    drop_tiny_contours(cs)
    ax.clabel(cs, fmt="%d", fontsize=8, colors="white", inline=True, inline_spacing=4)
    return cs


def add_practice_note(fig, y=0.025):
    """The confound belongs on the figure, not just in the README: these windows
    straddle both a satellite upgrade and a change in how plumes are drawn."""
    fig.text(
        y=y, x=0.5,
        s="Analyst practice changed between these windows (GOES-16/17 from 2017–2018; "
          "fewer, much larger polygons after ~2020).\nDay counts are less sensitive to "
          "that than area is, but the difference is not a clean physical trend.",
        ha="center", va="bottom", fontsize=8.5, color=MUTED, linespacing=1.5,
    )


def style_colorbar(fig, mesh, cax, label, extend):
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=0.8, color=MUTED, labelcolor=MUTED)
    return cb


def make_comparison(ds, spec, theme, periods, fields, means, lon, lat):
    """The two periods side by side, on one shared scale."""
    # vmax from a high percentile of the nonzero cells of both, so a saturated
    # corner of the late panel does not flatten the early one.
    pooled = np.concatenate([f.compressed() for f in fields])
    vmax = float(np.ceil(np.percentile(pooled, 99)))
    print(f"  shared color scale 0 -> {vmax:.0f} days/yr (99th pct of nonzero cells)")

    # Panel box is sized to the CONUS Lambert aspect (~2.1:1) so the maps sit snug
    # rather than letterboxed inside tall axes.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=DPI,
                             subplot_kw={"projection": PROJ})
    for ax, (y0, y1), field, mean in zip(axes, periods, fields, means):
        mesh = draw_map(ax, field, lon, lat, vmax, theme)
        ax.set_title(f"{y0}–{y1}", fontsize=13, color=INK, pad=22)
        ax.text(0.5, 1.025, f"CONUS mean {mean:.0f} days/yr", transform=ax.transAxes,
                ha="center", fontsize=9.5, color=INK_2)

    fig.suptitle(
        f"Average days per year under {spec.phrase}, contiguous United States",
        fontsize=14, color=INK, x=0.5, y=0.97,
    )
    fig.text(0.5, 0.917, spec.grid_note, ha="center", fontsize=9.5, color=INK_2)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.28, wspace=0.04)
    style_colorbar(fig, mesh, fig.add_axes([0.30, 0.185, 0.40, 0.026]),
                   f"days per year with {spec.phrase} overhead", "max")
    add_practice_note(fig)

    out = FIG_DIR / f"{spec.key}_smoke_days_period_change.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def make_difference(ds, spec, theme, periods, diff, lon, lat):
    """Late minus early, one map.

    Scale choice is data-dependent and deliberate. When every cell moved the same
    way -- which is what the default periods give -- the field is sequential, so it
    gets the sequential ramp anchored at 0: light means "barely changed", dark means
    "changed a lot", and the ramp's whole range is spent on values that exist. Only
    a genuinely two-signed field gets the diverging treatment, symmetric about a
    neutral zero so equal gains and losses read equally.
    """
    (y0e, y1e), (y0l, y1l) = periods
    lo, hi = float(diff.min()), float(diff.max())
    two_signed = lo < 0 < hi

    if two_signed:
        lim = float(np.ceil(np.percentile(np.abs(diff.compressed()), 99)))
        cmap, vmin, vmax, extend = DIVERGING, -lim, lim, "both"
        print(f"  diverging scale {-lim:.0f} -> {lim:.0f} days/yr (both signs present)")
    else:
        vmax = float(np.ceil(np.percentile(np.abs(diff.compressed()), 99)))
        cmap, vmin, extend = theme.cmap, 0.0, "max"
        if lo < 0:  # every cell fell: same logic, mirrored
            cmap, vmin, vmax, extend = theme.cmap, -vmax, 0.0, "min"
        print(f"  sequential scale {vmin:.0f} -> {vmax:.0f} days/yr "
              f"(single-signed: every cell moved the same way)")

    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=DPI, subplot_kw={"projection": PROJ})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.20)
    mesh = draw_map(ax, diff, lon, lat, vmax, theme, cmap=cmap, vmin=vmin)
    draw_contours(ax, diff, lon, lat)

    fig.suptitle(
        f"Change in days per year under {spec.phrase}, contiguous United States",
        fontsize=14, color=INK, x=0.5, y=0.968,
    )
    fig.text(0.5, 0.915, f"average {y0l}–{y1l} minus average {y0e}–{y1e}",
             ha="center", fontsize=10.5, color=INK_2)

    # The headline of a difference map is the sign and the spread, and neither is
    # readable off the ramp -- so state them.
    direction = "increased" if lo >= 0 else "decreased" if hi <= 0 else "changed"
    span = (f"every CONUS cell {direction}, by "
            f"{min(abs(lo), abs(hi)):.0f} to {max(abs(lo), abs(hi)):.0f} days/yr"
            if not two_signed else
            f"CONUS range {lo:+.0f} to {hi:+.0f} days/yr")
    fig.text(0.5, 0.876, f"{span}; mean {diff.mean():+.0f}",
             ha="center", fontsize=9.5, color=INK_2)

    style_colorbar(fig, mesh, fig.add_axes([0.30, 0.115, 0.40, 0.026]),
                   f"change in days per year ({y0l}–{y1l} − {y0e}–{y1e})", extend)

    out = FIG_DIR / f"{spec.key}_smoke_days_period_difference.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--early", default="2006-2009", type=parse_period,
                    help="first period, YYYY-YYYY (default: 2006-2009)")
    ap.add_argument("--late", default="2021-2024", type=parse_period,
                    help="second period, YYYY-YYYY (default: 2021-2024)")
    ap.add_argument("--class", dest="cls", default="all", type=str.lower,
                    choices=sorted(SPECS), help="smoke class (default: all)")
    ap.add_argument("--cmap", default="gray", type=str.lower, choices=sorted(THEMES),
                    help="sequential ramp (default: gray)")
    args = ap.parse_args(argv)

    spec = SPECS[args.cls]
    theme = THEMES[args.cmap]
    src = IN_DIR / f"{spec.var}.nc"
    if not src.exists():
        print(f"ERROR: missing {src.name} in data/processed/.\n"
              f"       Run: python3 "
              f"{'grid_all_smoke.py' if spec.key == 'all' else 'process_smoke.py'}",
              file=sys.stderr)
        return 1

    ds = xr.open_dataset(src)
    lon, lat = ds["lon"].values, ds["lat"].values
    periods = [args.early, args.late]
    fields, means = zip(*(period_mean(ds, spec.var, p) for p in periods))
    diff = period_diff(ds, spec.var, *periods)

    out = make_comparison(ds, spec, theme, periods, fields, means, lon, lat)
    print(f"  wrote {out.relative_to(FIG_DIR.parent)}")
    out = make_difference(ds, spec, theme, periods, diff, lon, lat)
    print(f"  wrote {out.relative_to(FIG_DIR.parent)}")

    d0, d1 = means
    print(f"  {periods[0][0]}–{periods[0][1]}: {d0:.1f} days/yr   "
          f"{periods[1][0]}–{periods[1][1]}: {d1:.1f} days/yr   "
          f"({d1 / d0:.1f}x, {d1 - d0:+.1f} days/yr)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
