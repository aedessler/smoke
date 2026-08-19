#!/usr/bin/env python3
"""Figure 6 -- the MAIAC record against NOAA HMS heavy smoke.

Only the *heavy* density class appears anywhere in this figure. HMS density
classes are nested contours drawn by hand, and the outer envelope of any smoke
at all is delineated too inconsistently across the record to serve as a
reference; heavy smoke is also the regime in which MAIAC selects a smoke
aerosol model at all, so it is the like-for-like comparison.

(a) the two CONUS-aggregate monthly series on their own axes, and (b) all
cell-years on the common grid.

    python3 fig6_validation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import maiac_common as M
from maiac_common import C


def main() -> int:
    df = C.merged_monthly()
    grid = C.build_common_grid("heavy")
    cmask = C.conus_mask(grid).values
    hms = grid["hms_smoke_days"].values[:, cmask]
    mai = grid["maiac_smoke_days"].values[:, cmask]
    fig = plt.figure(figsize=(7.5, 6.6), dpi=M.DPI)
    # Two panels, not three: the 2x4 grid centres (b) under the full-width (a).
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.12], hspace=0.52,
                          wspace=0.34, left=0.085, right=0.955, top=0.90,
                          bottom=0.085)

    # --- (a) the two monthly series --------------------------------------
    ax = fig.add_subplot(gs[0, :])
    M.style_axes(ax)
    ax.fill_between(df["date"], 0, df["maiac_smoke_days"], color=M.SMOKE,
                    alpha=0.18, zorder=2, linewidth=0)
    ax.plot(df["date"], df["maiac_smoke_days"], color=M.SMOKE, linewidth=1.1,
            zorder=4, label="MAIAC smoke days")
    ax.set_ylabel("MAIAC smoke days per month", color=M.SMOKE, fontsize=8.4)
    ax.tick_params(axis="y", colors=M.SMOKE)
    ax.set_ylim(0, float(df["maiac_smoke_days"].max()) * 1.30)

    ax_r = ax.twinx()
    hms_area = df["hms_heavy_km2_days"] / 1e6
    ax_r.plot(df["date"], hms_area, color=M.HMS_C, linewidth=1.1, zorder=3,
              label="HMS heavy")
    ax_r.set_ylabel("HMS heavy, 10⁶ km² days", color=M.HMS_C, fontsize=8.4)
    ax_r.tick_params(axis="y", colors=M.HMS_C, length=3, width=0.8)
    # Each series is scaled to its own maximum: the panel compares the shape of
    # the two records, and neither axis is a claim about the other's units.
    ax_r.set_ylim(0, float(hms_area.max()) * 1.30)
    for s in ("top", "left", "bottom"):
        ax_r.spines[s].set_visible(False)
    ax_r.spines["right"].set_color(M.AXIS)
    ax_r.spines["right"].set_linewidth(0.8)

    stats = C.correlations(df["maiac_smoke_days"], df["hms_heavy_km2_days"])
    ax.text(0.014, 0.965,
            f"r = {stats['pearson_r']:.3f} over {stats['n']} months\n"
            f"Spearman ρ = {stats['spearman_r']:.3f}",
            transform=ax.transAxes, fontsize=7.5, color=M.INK_2, va="top",
            linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=M.SURFACE,
                      edgecolor=M.GRID, linewidth=0.7))
    ax.text(0, 1.155, "a   Two independent records of the same months",
            transform=ax.transAxes, fontsize=9.3, color=M.INK, va="bottom",
            fontweight="semibold")
    ax.text(0, 1.04,
            "Separate axes: the products are in different units and only the "
            "co-variation is being compared here.",
            transform=ax.transAxes, fontsize=7.7, color=M.INK_2, va="bottom")

    # --- (b) cell-years ---------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 1:3])
    M.style_axes(ax_b, grid_axis="both")
    ax_b.hexbin(hms.ravel(), mai.ravel(), gridsize=52, bins="log",
                cmap="Blues", mincnt=1, linewidths=0, zorder=3,
                extent=(0, 45, 0, 45))
    lim = 45
    ax_b.plot([0, lim], [0, lim], color=M.MUTED, linewidth=0.9,
              linestyle=(0, (4, 3)), zorder=5)
    rng = np.random.default_rng(0)
    sub = rng.choice(hms.size, 4000, replace=False)
    ts = C.theil_sen(hms.ravel()[sub], mai.ravel()[sub])
    xx = np.array([0, lim])
    ax_b.plot(xx, ts["intercept"] + ts["slope"] * xx, color=M.INK,
              linewidth=1.3, zorder=6)
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    ax_b.set_xlabel("HMS heavy smoke days", color=M.INK_2, fontsize=8.4)
    ax_b.set_ylabel("MAIAC smoke days", color=M.INK_2, fontsize=8.4)
    ax_b.text(0.045, 0.955,
              f"slope {ts['slope']:.2f}   offset {ts['intercept']:+.1f} d",
              transform=ax_b.transAxes, fontsize=7.4, color=M.INK, va="top")
    ax_b.text(0, 1.14, "b   Every cell, every year", transform=ax_b.transAxes,
              fontsize=9.3, color=M.INK, va="bottom", fontweight="semibold")
    ax_b.text(0, 1.03,
              "12,455 common cells × 14 years; dashed 1:1",
              transform=ax_b.transAxes, fontsize=7.6, color=M.INK_2, va="bottom")

    M.save(fig, "fig6_validation.png")
    plt.close(fig)
    grid.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
