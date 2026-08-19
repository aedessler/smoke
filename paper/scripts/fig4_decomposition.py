#!/usr/bin/env python3
"""Figure 4 -- more often, or thicker?

The archive satisfies

    smoke AOD index  =  smoke frequency  x  mean AOD when smoke is selected

exactly, so in logs the growth rate of the index is the sum of the growth rates
of the two factors and the split between them is a decomposition rather than
two loosely related numbers. All three panels are log axes for that reason: a
straight line is a constant proportional rate, and the vertical distance
between panels (a) and (c) is panel (b).

    python3 fig4_decomposition.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import maiac_common as M
from maiac_common import C


def panel(ax, yr, v, label, sub, unit, color):
    M.style_axes(ax)
    ax.set_yscale("log")
    ax.plot(yr, v, marker="o", markersize=3.4, linewidth=1.0, color=color,
            markerfacecolor=color, markeredgecolor="none", zorder=4)
    lt = M.log_trend(v, yr)
    # Theil-Sen through the logs, anchored at the geometric mean so the line
    # sits in the data rather than being extrapolated from an intercept.
    b = lt["percent_per_year"] / 100.0
    anchor = np.exp(np.mean(np.log(v)) - b * np.mean(yr))
    ax.plot(yr, anchor * np.exp(b * yr), color=M.INK, linewidth=1.3, zorder=5)

    ax.text(0, 1.20, label, transform=ax.transAxes, fontsize=9.2, color=M.INK,
            va="bottom", fontweight="semibold")
    ax.text(0, 1.055, sub, transform=ax.transAxes, fontsize=7.6,
            color=M.INK_2, va="bottom")
    ax.set_ylabel(unit, color=M.INK_2, fontsize=8.3)
    ax.set_xlim(2002.3, 2024.7)
    ax.set_xticks(np.arange(2004, 2025, 5))
    ax.grid(axis="y", which="minor", color=M.GRID, linewidth=0.4, zorder=0)
    ax.text(0.97, 0.055,
            f"{lt['percent_per_year']:+.1f} % yr$^{{-1}}$   "
            f"(×{lt['factor_over_window']:.1f} over 22 years)\n"
            f"p = {lt['kendall_p']:.3f}",
            transform=ax.transAxes, fontsize=7.4, color=M.INK_2, ha="right",
            va="bottom", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=M.SURFACE,
                      edgecolor=M.GRID, linewidth=0.7))
    return lt


def main() -> int:
    ann = M.window(M.annual_from_monthly(M.conus_monthly()))
    yr = ann["year"].values.astype(float)

    fig = plt.figure(figsize=(7.5, 6.9), dpi=M.DPI)
    gs = fig.add_gridspec(3, 1, hspace=0.60, left=0.10, right=0.975,
                          top=0.915, bottom=0.115)

    freq = panel(fig.add_subplot(gs[0]), yr, ann["smoke_days"].values,
                 "a   How often  —  smoke days per year",
                 "clear-sky smoke frequency, scaled to the length of each month",
                 "days", M.SMOKE)
    inten = panel(fig.add_subplot(gs[1]), yr, ann["smoke_intensity"].values,
                  "b   How thick  —  mean AOD when the smoke model is selected",
                  "ΣC / ΣD: the average 550 nm optical depth of a smoke-labelled pixel-day",
                  "AOD (550 nm)", "#8a6d3b")
    index = panel(fig.add_subplot(gs[2]), yr, ann["smoke_aod_index"].values,
                  "c   The two together  —  smoke AOD index",
                  "ΣC / ΣB: frequency × intensity, zero where there was no smoke",
                  "index", "#7c3f8f")

    share = freq["percent_per_year"] / (freq["percent_per_year"]
                                        + inten["percent_per_year"])
    fig.text(0.10, 0.028,
             f"(a) + (b) = {freq['percent_per_year']:.1f} + "
             f"{inten['percent_per_year']:.1f} = "
             f"{freq['percent_per_year'] + inten['percent_per_year']:.1f} % yr⁻¹ "
             f"against {index['percent_per_year']:.1f} % yr⁻¹ for (c). About "
             f"{share*100:.0f} % of the growth in\nthe index is smoke occurring "
             f"more often; the rest is smoke being thicker when it occurs.",
             fontsize=7.5, color=M.INK_2, ha="left", va="bottom", linespacing=1.6)

    M.save(fig, "fig4_decomposition.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
