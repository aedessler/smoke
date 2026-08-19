#!/usr/bin/env python3
"""Annual smoke frequency over CONUS from the MAIAC archive, as bars.

The MAIAC counterpart to ../noaa/plot_annual_bars.py. Covers the whole record,
2000-02 through 2025-07, which means the first and last years are partial and are
drawn as such -- see below, because for this variable that matters a lot.

WHICH NUMBER THIS IS. The HMS figure sums analyst-drawn smoke AREA over each year.
MAIAC has no polygon area, but it has something with the same shape: the count of
1 km pixel-days on which the retrieval selected a smoke aerosol model. Divided by
the valid pixel-day count it becomes a frequency -- the share of everything MODIS
successfully saw that was smoky -- which is the quantity modis/README.md names for
comparing against the HMS annual series.

The raw smoke pixel-day count is deliberately NOT plotted: valid observations
roughly double in mid-2002 when Aqua joins Terra, so a raw count would show a step
that is a change in the satellite fleet, not in the smoke. The ratio divides that
out. It cannot divide out everything -- Aqua's afternoon overpass sees a different
part of the diurnal cycle -- so 2000 through mid-2002 is Terra-only and is flagged
on the figure.

AGGREGATION. Ratio of sums, never mean of ratios (modis/README.md, caveat 2):
sum both weights over all cells and all months of the year, then divide. This is
also why no minimum-weight mask is applied by default: summing the weights already
weights each cell by how much MODIS actually saw there, so a thin cell cannot shout
as loudly as a well-observed one. --min-weight is there for sensitivity tests.

Run from the modis/ folder (paths resolve relative to it either way):

    python3 maiac/plot_annual_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[1]  # the modis/ folder
DEFAULT_NC = ROOT / "data" / "maiac_smoke_25km_monthly.nc"
FIG_DIR = ROOT / "figures"
OUT_CSV = ROOT / "data" / "annual_maiac_smoke_frequency.csv"

# Design tokens, matching the HMS figures by eye. Duplicated rather than imported:
# the top-level README makes it a rule that the two products share no code, so a
# tokens import across the folder boundary would be the first violation of it.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_1_PALE = "#a9cbf0"

DPI = 150
AQUA_START = "2002-07"  # Aqua joins Terra; observations per pixel-day roughly double

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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def annual_table(ds, min_weight: float = 0.0) -> pd.DataFrame:
    """Per-year smoke frequency, plus what it took to get there.

    `months` and `span` exist so a partial year can be drawn as a partial year
    rather than silently compared against full ones.
    """
    smoke = ds["smoke_pixel_day_weight"]
    valid = ds["valid_pixel_day_weight"]

    # The smoke-AOD accumulator (sum of AOD over smoke pixel-days) is not stored,
    # but smoke_aod_index is exactly that sum over the valid weight, so multiplying
    # it back out recovers it -- verified against mean_smoke_aod_055 x smoke weight
    # to float32 noise. Recovering it from the INDEX rather than from the
    # conditional mean is the point: the index is 0 where nothing burned, whereas
    # the conditional mean is NaN there, and building a total out of a field that
    # goes missing on the clean cases is precisely the trap in README caveat 1.
    aod_sum = (ds["smoke_aod_index"] * valid).fillna(0.0)

    if min_weight > 0:
        thin = valid < min_weight
        smoke = smoke.where(~thin, 0.0)
        aod_sum = aod_sum.where(~thin, 0.0)
        valid = valid.where(~thin, 0.0)

    year = ds["time"].dt.year
    smoke_yr = smoke.sum(("y", "x")).groupby(year).sum()
    valid_yr = valid.sum(("y", "x")).groupby(year).sum()
    aod_yr = aod_sum.sum(("y", "x")).groupby(year).sum()
    months = ds["time"].dt.month.groupby(year)

    df = pd.DataFrame({
        "year": [int(y) for y in smoke_yr["year"].values],
        "smoke_pixel_days": smoke_yr.values,
        "valid_pixel_days": valid_yr.values,
        "smoke_aod_sum": aod_yr.values,
    })
    covered = {int(y): sorted(int(m) for m in g.values) for y, g in months}
    df["months"] = [len(covered[y]) for y in df["year"]]
    df["span"] = [
        f"{MONTHS[covered[y][0] - 1]}–{MONTHS[covered[y][-1] - 1]}"
        if len(covered[y]) < 12 else "Jan–Dec"
        for y in df["year"]
    ]
    # Three quantities, one identity: index = frequency x intensity, exactly, because
    # all three are ratios of the same sums. Every one is a ratio of sums over the
    # whole year, never a mean of monthly or per-cell ratios (README caveat 2).
    df["smoke_frequency_pct"] = 100 * df["smoke_pixel_days"] / df["valid_pixel_days"]
    df["smoke_aod_index"] = df["smoke_aod_sum"] / df["valid_pixel_days"]
    df["mean_smoke_aod"] = df["smoke_aod_sum"] / df["smoke_pixel_days"]
    return df


def style_axes(ax):
    """Recessive chrome: no top/right spines, hairline grid behind the bars."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def bars(ax, df, column):
    """Bars for one column, with partial years hatched rather than drawn as peers."""
    partial = df["months"].values < 12
    v = df[column].values
    ax.bar(df["year"][~partial], v[~partial], color=SERIES_1, width=0.72,
           zorder=3, linewidth=0)
    ax.bar(df["year"][partial], v[partial], color=SERIES_1_PALE, width=0.72,
           zorder=3, linewidth=0.8, edgecolor=SERIES_1, hatch="///")
    ax.set_ylim(bottom=0)


def make_decomposition(df: pd.DataFrame, out: Path) -> Path:
    """The index and the two factors it is the product of.

    One panel each, stacked, rather than two scales on one pair of axes: the three
    quantities have unrelated units and a twin y-axis would invite reading a
    crossover between them as meaning something.

    The panel order is the argument. The index is what "how smoky was that year"
    should mean, and the two below it say which way it got there -- 2017 and 2021
    reach similar indices by opposite routes.
    """
    panels = [
        ("smoke_aod_index", "smoke AOD index",
         "how smoky, all in — mean 550 nm AOD contributed by smoke retrievals,\n"
         "spread over every valid observation"),
        ("smoke_frequency_pct", "% of valid pixel-days",
         "how often — share of valid observations retrieved with a smoke model"),
        ("mean_smoke_aod", "mean AOD when smoky",
         "how thick — mean 550 nm AOD over just the smoke pixel-days"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.6), dpi=DPI, sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for ax, (column, ylabel, note) in zip(axes, panels):
        style_axes(ax)
        bars(ax, df, column)
        ax.set_ylabel(ylabel, color=INK_2)
        ax.text(0, 1.03, note, transform=ax.transAxes, fontsize=9, color=INK_2,
                va="bottom", linespacing=1.4)

    # Only the top panel carries the partial-year callouts; three copies of the
    # same note would be noise.
    for row in df[df["months"].values < 12].itertuples():
        axes[0].annotate(f"{row.span} only", xy=(row.year, row.smoke_aod_index),
                         xytext=(0, 5), textcoords="offset points", ha="center",
                         fontsize=8, color=INK_2)

    axes[-1].set_xticks(df["year"][::2])
    axes[-1].margins(x=0.02)

    fig.suptitle("Annual smoke over the contiguous United States, decomposed",
                 fontsize=13.5, color=INK, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.955,
             "MAIAC MCD19A2.061, "
             f"{df['year'].min()}–{df['year'].max()}. The top panel is the product "
             "of the bottom two, exactly — all three are ratios of the same sums. "
             "Hatched years are partial.",
             fontsize=9.5, color=INK_2)

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def make_figure(df: pd.DataFrame, out: Path) -> Path:
    full = df[df["months"] == 12]
    y = df["smoke_frequency_pct"].values
    med = float(full["smoke_frequency_pct"].median())

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=DPI)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)

    # Partial years get a hatched, pale bar. They are not comparable to the full
    # ones -- smoke is a summer-autumn signal, so a year missing August onward is
    # missing most of what the bar is supposed to measure -- and drawing them the
    # same as the rest would invite exactly that comparison.
    partial = df["months"].values < 12
    ax.bar(df["year"][~partial], y[~partial], color=SERIES_1, width=0.72,
           zorder=3, linewidth=0)
    ax.bar(df["year"][partial], y[partial], color=SERIES_1_PALE, width=0.72,
           zorder=3, linewidth=0.8, edgecolor=SERIES_1, hatch="///")
    for row in df[partial].itertuples():
        ax.annotate(f"{row.span} only", xy=(row.year, row.smoke_frequency_pct),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_2)

    ax.axhline(med, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    # Park the median label over the shortest bar, where there is guaranteed room
    # above the line. At the left edge, which is where the HMS figure puts it, it
    # collides with the 2000 bar and its partial-year note.
    quiet = int(df.loc[df["smoke_frequency_pct"].idxmin(), "year"])
    ax.annotate(f"median {med:.1f}%", xy=(quiet, med), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)

    ax.set_ylabel("% of valid pixel-days under smoke", color=INK_2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(df["year"][::2])
    ax.margins(x=0.02)
    ax.set_title(
        "Annual smoke frequency over the contiguous United States",
        fontsize=13.5, color=INK, pad=44, loc="left",
    )
    ax.text(
        0, 1.075,
        f"Share of valid MODIS observations retrieved with a smoke aerosol model. "
        f"MAIAC MCD19A2.061, {df['year'].min()}–{df['year'].max()}.",
        transform=ax.transAxes, fontsize=9.5, color=INK_2,
    )
    ax.text(
        0, 1.022,
        f"Hatched years are partial. Before {AQUA_START} the record is Terra-only; "
        f"Aqua roughly doubles observations per pixel-day from then on.",
        transform=ax.transAxes, fontsize=8.5, color=MUTED,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("netcdf", nargs="?", default=str(DEFAULT_NC),
                    help=f"MAIAC archive (default: {DEFAULT_NC.relative_to(ROOT)})")
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="drop cell-months with fewer valid pixel-days (default: 0, "
                         "i.e. keep all -- the weighted sum already handles thin cells)")
    ap.add_argument("-o", "--out", default="",
                    help="output PNG (default: figures/annual_maiac_smoke_frequency.png)")
    args = ap.parse_args(argv)

    src = Path(args.netcdf)
    if not src.exists():
        print(f"ERROR: {src} not found.\n"
              f"       The archive is built by maiac/concat_archive.py from "
              f"data/maiac/monthly/.", file=sys.stderr)
        return 1

    ds = xr.open_dataset(src)
    df = annual_table(ds, args.min_weight)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV.relative_to(ROOT)} ({len(df)} rows)")

    out = Path(args.out) if args.out else FIG_DIR / "annual_maiac_smoke_frequency.png"
    make_figure(df, out)
    print(f"  wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")

    dec = FIG_DIR / "annual_maiac_smoke_decomposition.png"
    make_decomposition(df, dec)
    print(f"  wrote {dec.relative_to(ROOT)}")

    full = df[df["months"] == 12].sort_values("smoke_frequency_pct", ascending=False)
    top = ", ".join(f"{int(r.year)} ({r.smoke_frequency_pct:.1f}%)"
                    for r in full.head(4).itertuples())
    print(f"  highest complete years: {top}")
    partial = df[df["months"] < 12]
    if len(partial):
        print("  partial: " + ", ".join(
            f"{int(r.year)} ({r.span}, {r.months} months)" for r in partial.itertuples()))
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
