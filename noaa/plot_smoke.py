#!/usr/bin/env python3
"""Visualize the processed NOAA HMS smoke products.

Which smoke class to plot:

    python3 plot_smoke.py                # uses DEFAULT_CLASS below
    python3 plot_smoke.py --class all    # any density, 2006-2025
    python3 plot_smoke.py --class heavy

"all" is the all-density series. It needs no density filtering, so it runs five
years longer than the density classes (2006 vs 2011) -- that longer record is the
main reason to use it. process_smoke.py only grids the one class named by its own
SMOKE_CLASS, so the all-density maps need all_smoke_days.nc from grid_all_smoke.py;
without it "all" still produces its time series and the maps are skipped with a
message.

For a density class, <cls>_smoke_days.nc must exist, i.e. process_smoke.py must
have been run with the matching SMOKE_CLASS.

Reads only data/processed/ (no network). Writes to figures/:

  <cls>_smoke_days_panel.png       facet of annual maps          (needs the .nc)
  maps/<cls>_smoke_days_YYYY.png   the same map, one per year    (needs the .nc)
  monthly_<cls>_smoke_area.png     monthly km^2*days
  annual_<cls>_smoke_area.png      annual totals
  <cls>_seasonality_and_trend.png  monthly climatology + annual totals

and to data/processed/:

  annual_<cls>_smoke_area.csv      the annual table behind the annual figure
  monthly_all_smoke_area.csv       "all" only -- aggregated from the daily table,
                                   since process_smoke.py writes no all-density file

Note: HMS density classes are nested, so Light is the OUTER envelope -- light-class
figures are near-identical to "all" over 2011-2025, not a light-only band.

The maps take --cmap blue (default) or --cmap gray; each ramp carries its own
geography-line colors, since chrome that reads over blue does not read over gray.

Run: python3 plot_smoke.py [--class all|light|medium|heavy] [--cmap blue|gray]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

# Used when --class is not given. Must name a class process_smoke.py has gridded,
# unless it is "all" (which needs only the daily table).
DEFAULT_CLASS = "Light"

ALL_SMOKE_START_YEAR = 2006  # 2005 is partial (record starts 2005-08-05)

# --- design tokens (light surface) -------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"  # categorical slot 1 (blue)
SERIES_2 = "#eb6834"  # categorical slot 2 (orange)

# Sequential blue ramp, steps 100 -> 700.
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP.set_bad(SURFACE)  # outside CONUS

# Sequential gray ramp, same 13 lightness steps carrying the surface's warmth so it
# sits with the rest of the chrome rather than reading as a cold second neutral.
SEQ_GRAY = [
    "#e4e3df", "#d5d4cf", "#c5c4be", "#b4b3ad", "#a3a29c", "#92918b",
    "#82817b", "#71706b", "#605f5a", "#50504b", "#41403c", "#32312e", "#232220",
]
CMAP_GRAY = LinearSegmentedColormap.from_list("seq_gray", SEQ_GRAY)
CMAP_GRAY.set_bad(SURFACE)

DPI = 150

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "font.size": 9,
    }
)


# ---------------------------------------------------------------- class spec

@dataclass(frozen=True)
class Spec:
    """Everything that varies between the smoke classes.

    `phrase` slots into "area under {phrase}"; `attr` into "daily {attr} area";
    `noun` starts a sentence. Keeping the three separate avoids the awkward
    "area under all smoke" / "Daily all-smoke area" phrasings.
    """

    key: str  # filename stem: all | light | medium | heavy
    phrase: str
    attr: str
    noun: str
    gridded: bool  # process_smoke.py can write this class's grid, so require it

    @property
    def var(self) -> str:
        return f"{self.key}_smoke_days"

    @property
    def has_grid(self) -> bool:
        """Whether the gridded file is actually on disk.

        For "all" the grid is optional -- it comes from grid_all_smoke.py, which is
        a separate run -- so the maps are drawn if it exists and skipped if not.
        """
        return (IN_DIR / f"{self.var}.nc").exists()

    @property
    def grid_note(self) -> str:
        """Subtitle line naming the polygon filter behind the maps."""
        if self.key == "all":
            return ("NOAA HMS smoke polygons of any density, including unclassified, "
                    "0.1° grid")
        return f"NOAA HMS smoke polygons classified Density = {self.key.title()}, 0.1° grid"


SPECS = {
    "all": Spec("all", "smoke of any density", "smoke", "Smoke of any density", False),
    "light": Spec("light", "light smoke", "light-smoke", "Light smoke", True),
    "medium": Spec("medium", "medium smoke", "medium-smoke", "Medium smoke", True),
    "heavy": Spec("heavy", "heavy smoke", "heavy-smoke", "Heavy smoke", True),
}


@dataclass(frozen=True)
class MapTheme:
    """A map ramp plus the geography-line colors that stay legible on top of it.

    The line colors are not decoration: they have to hold at every step of the
    ramp. Over the blue ramp the warm-gray chrome separates by hue, so it reads at
    any value. An achromatic ramp takes that away -- AXIS and MUTED land in the
    middle of the gray scale and state borders disappear under mid-range cells --
    so the gray theme puts the internal borders in the surface color, which
    separates by lightness against everything but the palest cells, and darkens
    the outline, which sits on masked (surface-colored) ground anyway.
    """

    cmap: LinearSegmentedColormap
    states: str
    outline: str


THEMES = {
    "blue": MapTheme(CMAP, AXIS, MUTED),
    "gray": MapTheme(CMAP_GRAY, SURFACE, INK_2),
}


def style_axes(ax, grid_axis="y"):
    """Recessive chrome: no top/right spines, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ---------------------------------------------------------------- load

def load_monthly(spec: Spec) -> pd.DataFrame:
    """Monthly smoke-area-days as columns year, month, t, km2_days.

    Density classes read the monthly file process_smoke.py wrote. "all" is
    aggregated here from the daily table, over a complete year x month grid so
    smoke-free months are explicit zeros rather than missing rows.
    """
    if spec.key != "all":
        df = pd.read_csv(IN_DIR / f"monthly_{spec.key}_smoke_area.csv")
        df = df.rename(columns={f"{spec.key}_km2_days": "km2_days",
                                f"n_days_with_{spec.key}": "n_days"})
    else:
        daily = pd.read_csv(IN_DIR / "daily_smoke_area.csv", parse_dates=["date"])
        daily = daily[daily["date"].dt.year >= ALL_SMOKE_START_YEAR]
        grouped = daily.groupby(
            [daily["date"].dt.year.rename("year"),
             daily["date"].dt.month.rename("month")]
        ).agg(km2_days=("all_km2", "sum"),
              n_days=("all_km2", lambda s: int((s > 0).sum())))
        years = range(ALL_SMOKE_START_YEAR, int(daily["date"].dt.year.max()) + 1)
        full = pd.MultiIndex.from_product([years, range(1, 13)],
                                          names=["year", "month"])
        df = grouped.reindex(full, fill_value=0).reset_index()
        out = IN_DIR / "monthly_all_smoke_area.csv"
        df.rename(columns={"km2_days": "all_km2_days"}).to_csv(out, index=False)
        print(f"  wrote {out.name} ({len(df)} rows)")

    df["t"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    return df.sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------- maps

def draw_map(ax, field, lon, lat, vmax, theme: MapTheme, cmap=None, vmin=0):
    """One CONUS choropleth of smoke days.

    `cmap`/`vmin` override the theme's ramp and floor for fields that are not
    counts-from-zero (a difference map, say); the geography lines still come from
    the theme, since those are what keep the map readable.
    """
    ax.set_extent([-125, -66, 23.5, 50], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(
        lon, lat, field, cmap=cmap if cmap is not None else theme.cmap,
        vmin=vmin, vmax=vmax,
        shading="nearest", transform=ccrs.PlateCarree(), zorder=1,
    )
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor=theme.states, linewidth=0.3, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor=theme.outline, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor=theme.outline, linewidth=0.5, zorder=2)
    ax.spines["geo"].set_visible(False)
    return mesh


def masked_field(ds, spec: Spec, year):
    """Smoke days for one year, masked outside CONUS and at true zeros.

    Masking the zeros matters: the lightest ramp step is still visibly blue, so
    leaving them in tints the whole country and low counts stop being readable.
    Zero smoke days is genuinely "nothing to draw" — the state outlines
    still carry the geography.
    """
    inside = ds["in_conus"].values.astype(bool)
    field = ds[spec.var].sel(year=year).values.astype(float)
    return np.ma.masked_where(~inside | (field == 0), field)


def make_maps(ds, spec: Spec, theme: MapTheme):
    lon, lat = ds["lon"].values, ds["lat"].values
    years = [int(y) for y in ds["year"].values]

    # Shared scale across every panel so years are directly comparable. vmax from a
    # high percentile of nonzero cells, so one extreme year doesn't flatten the rest.
    inside = ds["in_conus"].values.astype(bool)
    allvals = ds[spec.var].values[:, inside]
    nonzero = allvals[allvals > 0]
    vmax = float(np.ceil(np.percentile(nonzero, 99)))
    print(f"  shared color scale 0 -> {vmax:.0f} days (99th pct of nonzero cells)")

    # --- facet panel, sized to fit however many years the file holds
    ncol = 4
    nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 2.15 * nrow), dpi=DPI, squeeze=False,
        subplot_kw={"projection": ccrs.LambertConformal(
            central_longitude=-96, standard_parallels=(33, 45))},
    )
    for ax, year in zip(axes.ravel(), years):
        mesh = draw_map(ax, masked_field(ds, spec, year), lon, lat, vmax, theme)
        ax.set_title(str(year), fontsize=11, color=INK, pad=4)
    for ax in axes.ravel()[len(years):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Days per year under {spec.phrase}, contiguous United States",
        fontsize=14, color=INK, x=0.5, y=0.975,
    )
    fig.text(
        0.5, 0.935, spec.grid_note,
        ha="center", fontsize=9.5, color=INK_2,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10,
                        wspace=0.03, hspace=0.15)
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.018])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cb.set_label(f"days with {spec.phrase} overhead", color=INK_2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=0.8, color=MUTED, labelcolor=MUTED)

    out = FIG_DIR / f"{spec.key}_smoke_days_panel.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")

    # --- one file per year, same scale
    map_dir = FIG_DIR / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        fig, ax = plt.subplots(
            figsize=(7.2, 4.6), dpi=DPI,
            subplot_kw={"projection": ccrs.LambertConformal(
                central_longitude=-96, standard_parallels=(33, 45))},
        )
        mesh = draw_map(ax, masked_field(ds, spec, year), lon, lat, vmax, theme)
        ax.set_title(f"Days under {spec.phrase} — {year}",
                     fontsize=12.5, color=INK, pad=6)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.20)
        cax = fig.add_axes([0.30, 0.12, 0.40, 0.028])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
        cb.set_label(f"days with {spec.phrase} overhead", color=INK_2, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=3, width=0.8, color=MUTED, labelcolor=MUTED)
        fig.savefig(map_dir / f"{spec.key}_smoke_days_{year}.png", dpi=DPI)
        plt.close(fig)
    print(f"  wrote {len(years)} per-year maps to figures/maps/")


# ---------------------------------------------------------------- time series

def make_monthly_series(monthly, spec: Spec):
    y = monthly["km2_days"] / 1e6  # million km^2 * days
    y0, y1 = int(monthly.year.min()), int(monthly.year.max())

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=DPI)
    style_axes(ax)
    ax.fill_between(monthly["t"], y, color=SERIES_1, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(monthly["t"], y, color=SERIES_1, linewidth=2, zorder=3)

    # Direct-label only the few months that carry the story, and keep the labels
    # from colliding by requiring 6 months of separation between them.
    labeled = []
    for row in monthly.sort_values("km2_days", ascending=False).itertuples():
        if len(labeled) >= 3:
            break
        if any(abs((row.t - t).days) < 183 for t in labeled):
            continue
        labeled.append(row.t)
        ax.annotate(
            f"{row.t:%b %Y}",
            xy=(row.t, row.km2_days / 1e6),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=9, color=INK_2,
        )

    ax.set_ylabel("million km² · days", color=INK_2)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.set_title(
        f"Monthly area under {spec.phrase} over the contiguous United States",
        fontsize=13.5, color=INK, pad=30, loc="left",
    )
    ax.text(
        0, 1.025,
        f"Daily {spec.attr} area summed over each month. NOAA HMS, {y0}–{y1}.",
        transform=ax.transAxes, fontsize=9.5, color=INK_2,
    )
    fig.tight_layout()
    out = FIG_DIR / f"monthly_{spec.key}_smoke_area.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def make_annual_series(monthly, spec: Spec):
    """Standalone annual totals, plus the annual table as CSV.

    Deliberately no fitted trend line. The polygon-granularity shift documented in
    the README (analysts moved to fewer, much larger polygons after ~2020) inflates
    the recent end of this series, so a regression through it would imply a physical
    trend the data cannot support. The peak-vs-median reference band is descriptive.
    """
    ann = monthly.groupby("year").agg(
        km2_days=("km2_days", "sum"),
        n_days=("n_days", "sum"),
        peak_month_km2_days=("km2_days", "max"),
    )
    ann["mean_daily_km2"] = ann["km2_days"] / [
        366 if y % 4 == 0 else 365 for y in ann.index
    ]
    ann["peak_month"] = (
        monthly.loc[monthly.groupby("year")["km2_days"].idxmax()]
        .set_index("year")["month"]
    )
    out_csv = IN_DIR / f"annual_{spec.key}_smoke_area.csv"
    ann.reset_index().to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.name} ({len(ann)} rows)")

    y = ann["km2_days"] / 1e6
    med = float(y.median())

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    style_axes(ax)

    # Median reference, so the recent years read against a baseline rather than
    # against the eye's guess at the middle of the axis.
    ax.axhline(med, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"median {med:,.0f}", xy=(ann.index[0], med), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=MUTED)

    ax.fill_between(ann.index, y, color=SERIES_1, alpha=0.13, linewidth=0, zorder=2)
    ax.plot(ann.index, y, color=SERIES_1, linewidth=2, zorder=3)
    ax.scatter(ann.index, y, s=26, color=SERIES_1, zorder=4,
               edgecolor=SURFACE, linewidth=1.2)

    peak = int(y.idxmax())
    ax.scatter([peak], [y[peak]], s=46, color=SERIES_2, zorder=5,
               edgecolor=SURFACE, linewidth=1.2)
    ax.annotate(f"{peak}\n{y[peak]:,.0f}", xy=(peak, y[peak]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=SERIES_2, linespacing=1.3)

    ax.set_ylabel("million km² · days", color=INK_2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(ann.index[::2])
    ax.margins(x=0.03)
    ax.set_title(
        f"Annual area under {spec.phrase} over the contiguous United States",
        fontsize=13.5, color=INK, pad=30, loc="left",
    )
    ax.text(
        0, 1.025,
        f"Daily {spec.attr} area summed over each year. "
        f"NOAA HMS, {int(ann.index.min())}–{int(ann.index.max())}.",
        transform=ax.transAxes, fontsize=9.5, color=INK_2,
    )
    fig.tight_layout()
    out = FIG_DIR / f"annual_{spec.key}_smoke_area.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def make_seasonality_and_trend(monthly, spec: Spec):
    y0, y1 = int(monthly.year.min()), int(monthly.year.max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=DPI,
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- left: monthly climatology with interquartile spread
    style_axes(ax1)
    g = monthly.groupby("month")["km2_days"]
    med = g.median() / 1e6
    q1, q3 = g.quantile(0.25) / 1e6, g.quantile(0.75) / 1e6
    months = np.arange(1, 13)
    ax1.fill_between(months, q1, q3, color=SERIES_1, alpha=0.16,
                     linewidth=0, zorder=2)
    ax1.plot(months, med, color=SERIES_1, linewidth=2, zorder=3)
    ax1.set_xticks(months)
    ax1.set_xticklabels(list("JFMAMJJASOND"))
    ax1.set_ylabel("million km² · days", color=INK_2)
    ax1.set_ylim(bottom=0)
    ax1.set_title("Seasonal cycle", fontsize=12, color=INK, loc="left", pad=26)
    ax1.text(0, 1.025, "median by calendar month, shaded 25th–75th percentile",
             transform=ax1.transAxes, fontsize=9, color=INK_2)

    # --- right: annual totals
    style_axes(ax2)
    ann = monthly.groupby("year")["km2_days"].sum() / 1e6
    ax2.bar(ann.index, ann.values, color=SERIES_1, width=0.68,
            zorder=3, linewidth=0)
    peak = ann.idxmax()
    ax2.bar([peak], [ann[peak]], color=SERIES_2, width=0.68, zorder=4, linewidth=0)
    ax2.annotate(f"{peak}", xy=(peak, ann[peak]), xytext=(0, 6),
                 textcoords="offset points", ha="center",
                 fontsize=9.5, color=SERIES_2)
    ax2.set_ylabel("million km² · days", color=INK_2)
    ax2.set_xticks(ann.index[::2])
    ax2.set_title("Annual total", fontsize=12, color=INK, loc="left", pad=26)
    ax2.text(0, 1.025, f"sum of daily {spec.attr} area over the year",
             transform=ax2.transAxes, fontsize=9, color=INK_2)

    fig.suptitle(f"{spec.noun} over the contiguous United States, {y0}–{y1}",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / f"{spec.key}_seasonality_and_trend.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--class", dest="cls", default=DEFAULT_CLASS, type=str.lower,
        choices=sorted(SPECS), help=f"smoke class to plot (default: {DEFAULT_CLASS})",
    )
    ap.add_argument(
        "--cmap", default="blue", type=str.lower, choices=sorted(THEMES),
        help="sequential ramp for the maps (default: blue)",
    )
    args = ap.parse_args(argv)
    spec = SPECS[args.cls]
    theme = THEMES[args.cmap]
    print(f"Plotting '{spec.key}' smoke ...")

    # Preflight: process_smoke.py writes products for one class per run, so asking
    # for a class it has not processed is the easy mistake. Say so plainly instead
    # of failing with a FileNotFoundError traceback partway through.
    missing = [
        p.name
        for p in ([IN_DIR / f"monthly_{spec.key}_smoke_area.csv"] if spec.key != "all"
                  else [IN_DIR / "daily_smoke_area.csv"])
        + ([IN_DIR / f"{spec.var}.nc"] if spec.gridded else [])
        if not p.exists()
    ]
    if missing:
        print(
            f"ERROR: missing {', '.join(missing)} in data/processed/.\n"
            f"       Re-run process_smoke.py with SMOKE_CLASS = \"{spec.key.title()}\".",
            file=sys.stderr,
        )
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly(spec)

    if spec.gridded or spec.has_grid:
        print("Maps ...")
        make_maps(xr.open_dataset(IN_DIR / f"{spec.var}.nc"), spec, theme)
    else:
        # process_smoke.py grids only its own SMOKE_CLASS, so the all-density
        # day-count field is a separate, optional run.
        print("Maps ... skipped (no all_smoke_days.nc; run grid_all_smoke.py to add it)")

    print("Time series ...")
    make_monthly_series(monthly, spec)
    make_annual_series(monthly, spec)
    make_seasonality_and_trend(monthly, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
