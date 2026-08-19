#!/usr/bin/env python3
"""Figure 6 -- when in the year the smoke arrives, and where the growth went.

The mean seasonal cycle in each half of the record, so the change can be read
month by month. Because differences of means add to the annual difference,
the twelve gaps between the pairs are a complete accounting of the annual
change.

    python3 fig6_season.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import maiac_common as M

EARLY_END = 2013


def main() -> int:
    mon = M.window(M.conus_monthly())
    early = mon[mon.year <= EARLY_END].groupby("month")["smoke_days"].mean()
    late = mon[mon.year > EARLY_END].groupby("month")["smoke_days"].mean()

    fig = plt.figure(figsize=(7.5, 3.3), dpi=M.DPI)
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.965, top=0.80, bottom=0.14)

    # --- the seasonal cycle, halved -------------------------------------
    ax = fig.add_subplot(gs[0])
    M.style_axes(ax)
    x = np.arange(1, 13)
    ax.bar(x - 0.19, early.values, width=0.36, color="#b9cfe9",
           edgecolor="none", zorder=3, label=f"{M.FIRST_YEAR}–{EARLY_END}")
    ax.bar(x + 0.19, late.values, width=0.36, color=M.SMOKE, edgecolor="none",
           zorder=3, label=f"{EARLY_END + 1}–{M.LAST_YEAR}")
    ax.set_xticks(x)
    ax.set_xticklabels(M.MONTH_ABBR, fontsize=8)
    ax.set_ylabel("mean smoke days", color=M.INK_2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", handlelength=1.2,
              labelspacing=0.35)

    change = (late - early)
    jul_sep = float(change.loc[7:9].sum() / change.sum())
    ax.annotate(f"{jul_sep*100:.0f} % of the rise\nis July–September",
                xy=(8.19, late[8]), xytext=(10.6, late[8] * 0.94),
                fontsize=7.4, color=M.INK_2, ha="center", va="top",
                linespacing=1.6,
                arrowprops=dict(arrowstyle="-", linewidth=0.6, color=M.MUTED,
                                shrinkA=2, shrinkB=3))
    ax.text(0, 1.155, "The seasonal cycle, first half against second",
            transform=ax.transAxes, fontsize=9.3, color=M.INK, va="bottom",
            fontweight="semibold")
    ax.text(0, 1.045,
            "Averages of eleven years each. Differences of means add to the "
            "annual change, so this is a full accounting of it.",
            transform=ax.transAxes, fontsize=7.7, color=M.INK_2, va="bottom")

    M.save(fig, "fig6_season.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
