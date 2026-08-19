#!/usr/bin/env python3
"""Figure 5 -- where the smoke is, and where it is increasing.

(a) the 22-year mean, (b) the per-cell Theil-Sen slope with the cells that
reach local significance stippled, and (c) the difference between the two
halves of the record.

The stippling in (b) marks cells whose rank correlation with time beats the
95th percentile of 2,000 year-label shuffles of the whole field. Shuffling the
labels leaves the spatial correlation of the field intact, which a per-cell
p-value cannot do -- neighbouring 25 km cells see the same plumes.

    python3 fig5_maps.py
"""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

import maiac_common as M
from maiac_common import C

ALBERS = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23,
                              standard_parallels=(29.5, 45.5))
EXTENT = [-2.30e6, 2.35e6, 2.0e5, 3.25e6]


def basemap(ax):
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.25,
                   edgecolor="#b8b7ae", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4,
                   edgecolor="#8d8c84", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4,
                   edgecolor="#8d8c84", zorder=4)
    ax.set_extent(EXTENT, crs=ALBERS)
    ax.spines["geo"].set_visible(False)


def mapped(fig, slot, edges, field, cmap, vmin, vmax, title, note, label,
           extend="max"):
    ax = fig.add_subplot(slot, projection=ALBERS)
    basemap(ax)
    im = ax.pcolormesh(edges[0], edges[1], field, cmap=cmap, vmin=vmin,
                       vmax=vmax, transform=ALBERS, zorder=3, shading="flat",
                       rasterized=True)
    ax.text(0.005, 1.115, title, transform=ax.transAxes, fontsize=9.2,
            color=M.INK, va="bottom", fontweight="semibold")
    ax.text(0.005, 1.015, note, transform=ax.transAxes, fontsize=7.6,
            color=M.INK_2, va="bottom")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.052,
                      pad=0.03, extend=extend)
    cb.set_label(label, fontsize=7.7, color=M.INK_2, labelpad=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=0.7, color=M.MUTED,
                      labelcolor=M.MUTED)
    return ax


def main() -> int:
    ds = C.load_maiac()
    mask = M.domain(ds).values
    ann = M.annual_cell_days(ds)
    years = ann["year"].values.astype(float)
    vals = ann.values.reshape(ann.shape[0], -1)[:, mask.ravel()]
    slope = M.sen_slope(vals, years)
    field = M.field_test(vals, years)
    half = vals.shape[0] // 2
    diff = vals[half:].mean(0) - vals[:half].mean(0)

    def grid(v):
        g = np.full(mask.size, np.nan)
        g[mask.ravel()] = v
        return g.reshape(mask.shape)

    edges = C.cell_edges(ds)
    mean_map = grid(vals.mean(0))
    slope_map = grid(slope)
    diff_map = grid(diff)

    vmax = float(np.percentile(vals.mean(0), 99))
    smax = float(np.percentile(np.abs(slope), 99))
    dmax = float(np.percentile(np.abs(diff), 99))

    # Three maps, not four: a 2x4 grid lets the last one sit centred under the
    # pair above it rather than stranded in a corner.
    fig = plt.figure(figsize=(7.7, 6.9), dpi=M.DPI)
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.10,
                          left=0.02, right=0.98, top=0.915, bottom=0.055)

    # Light to dark, pale where there is little smoke: on a smoke map the
    # reader should not have to consult the colour bar to know which end is
    # which, which rules out a bright-at-both-ends ramp like magma.
    mapped(fig, gs[0, 0:2], edges, mean_map, "YlOrBr", 0, vmax,
           "a   Mean smoke days",
           f"{M.FIRST_YEAR}–{M.LAST_YEAR} average", "days per year")

    ax_b = mapped(fig, gs[0, 2:4], edges, slope_map, "RdBu_r", -smax, smax,
                  "b   Trend", "Theil–Sen slope; stipple: locally significant",
                  "days per year, per year", extend="both")
    sig = (field["p_cell"] < 0.05) & field["live"]
    xs, ys = np.meshgrid(ds["x"].values, ds["y"].values)
    xf = xs.ravel()[mask.ravel()][sig]
    yf = ys.ravel()[mask.ravel()][sig]
    # every second significant cell, so the stipple reads as texture rather
    # than as a second filled layer
    ax_b.plot(xf[::2], yf[::2], linestyle="none", marker=".", markersize=0.7,
              color="#1b1b1f", transform=ALBERS, zorder=5, alpha=0.85)

    mapped(fig, gs[1, 1:3], edges, diff_map, "RdBu_r", -dmax, dmax,
           "c   Second half minus first",
           "2014–2024 mean − 2003–2013 mean", "days per year", extend="both")

    M.save(fig, "fig5_maps.png")
    plt.close(fig)
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
