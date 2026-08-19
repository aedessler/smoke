#!/usr/bin/env python3
"""Figure 3 -- the record year by year, one map per year.

Figure 2 collapses the whole grid into a single number per year; this is the
same 22 years without that collapse. One small map each, all on one shared
colour scale, so the years are directly comparable and the reader can see what
a quiet year and a severe year actually look like on the ground.

Adapted from ../../modis/maiac/plot_annual_panel.py, which still lives there and
still draws the raw smoke frequency over the full 2000-2025 archive; the two are
deliberately different figures, not two copies of one. Here the quantity is
smoke days per year on the paper's analysis domain over the paper's window --
the same field Figure 5a averages -- so that the panels, the trend map and the
annual series are all the same number.

Two things follow from using this paper's definitions rather than the raw
frequency. Thin observation is handled by the domain rather than by a
per-year weight threshold: a cell is in or out for all 22 years, which is what
makes a per-cell comparison across years mean anything. And zeros are drawn,
not masked -- a zero here is "MODIS watched this cell all year and never
retrieved smoke", a real result in the quiet years, and hiding it would turn
2004 and 2010 into maps that look like missing data instead of clean air.

    python3 fig3_annual_panel.py
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
NCOL = 4
# Same ramp as Figure 5a, because it is the same quantity: pale where there is
# little smoke, so the reader does not have to consult the bar to know which
# end is which.
CMAP = "YlOrBr"


def main() -> int:
    ds = C.load_maiac()
    mask = M.domain(ds).values
    ann = M.annual_cell_days(ds)
    years = [int(y) for y in ann["year"].values]
    field = np.where(mask, ann.values, np.nan)

    # One scale for every panel or the years are not comparable, and vmax from
    # a high percentile rather than the maximum so that 2020 and 2021 cannot
    # flatten the other twenty.
    finite = field[np.isfinite(field)]
    vmax = float(np.ceil(np.percentile(finite, 99.5)))
    print(f"  shared colour scale 0 -> {vmax:.0f} days "
          f"(99.5th pct of cell-years; max {finite.max():.1f})")

    edges = C.cell_edges(ds)
    nrow = int(np.ceil(len(years) / NCOL))
    fig = plt.figure(figsize=(7.6, 1.30 * nrow + 0.55), dpi=M.DPI)
    gs = fig.add_gridspec(nrow, NCOL, hspace=0.10, wspace=0.03,
                          left=0.012, right=0.988, top=0.925, bottom=0.015)

    for k, (year, layer) in enumerate(zip(years, field)):
        ax = fig.add_subplot(gs[k // NCOL, k % NCOL], projection=ALBERS)
        im = ax.pcolormesh(edges[0], edges[1], layer, cmap=CMAP, vmin=0,
                           vmax=vmax, transform=ALBERS, zorder=3,
                           shading="flat", rasterized=True)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.12,
                       edgecolor="#b8b7ae", zorder=4)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.18,
                       edgecolor="#8d8c84", zorder=4)
        ax.set_extent(EXTENT, crs=ALBERS)
        ax.spines["geo"].set_visible(False)
        ax.text(0.5, 1.005, str(year), transform=ax.transAxes, fontsize=7.8,
                color=M.INK, ha="center", va="bottom")

    fig.text(0.012, 0.982, "Smoke days by year", fontsize=9.2, color=M.INK,
             va="bottom", fontweight="semibold")
    fig.text(0.012, 0.950,
             "Per-cell smoke days, every year of the analysis window on one "
             "colour scale", fontsize=7.6, color=M.INK_2, va="bottom")

    # The colour bar goes in the slots the last row leaves empty rather than
    # under the panels, which would cost a strip of height for nothing.
    empty = gs[nrow - 1, len(years) % NCOL:].get_position(fig)
    cax = fig.add_axes([empty.x0 + 0.16 * empty.width, empty.y0 + 0.46 * empty.height,
                        0.68 * empty.width, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("days per year", fontsize=7.7, color=M.INK_2, labelpad=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=0.7, color=M.MUTED,
                      labelcolor=M.MUTED)

    M.save(fig, "fig3_annual_panel.png")
    plt.close(fig)
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
