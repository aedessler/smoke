#!/usr/bin/env python3
"""Figure 2 -- the CONUS smoke record.

(a) every month of the archive, including the Terra-only years that are shown
    but excluded from every trend, and (b) the annual totals with the Theil-Sen
    fit, its confidence band, and the mean of each half of the record.

Panel (b) is drawn on a linear axis on purpose: the eye should see how much of
the record's mass sits in the last five years, which a log axis flattens away.

    python3 fig2_record.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import maiac_common as M
from maiac_common import C

EARLY_END = 2013
PALE = "#b9cfe9"
DEEP = "#1c4f8f"
# The half-means get their own hue and a solid, heavy stroke so that neither
# can be read as the fit or as the band around it.
MEAN_EARLY = "#7d8f3c"
MEAN_LATE = "#b4451c"


def main() -> int:
    mon = M.conus_monthly()
    ann = M.annual_from_monthly(mon)
    a = M.window(ann)
    yr = a["year"].values.astype(float)
    sd = a["smoke_days"].values

    fig = plt.figure(figsize=(7.4, 6.5), dpi=M.DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.44,
                          left=0.085, right=0.98, top=0.895, bottom=0.07)

    # --- (a) every month ------------------------------------------------
    ax = fig.add_subplot(gs[0])
    M.style_axes(ax)
    ax.axvspan(pd.Timestamp("2000-01-01"), pd.Timestamp("2002-12-31"),
               color="#f0efe8", zorder=0, linewidth=0)
    ax.fill_between(mon["date"], 0, mon["smoke_days"], color=M.SMOKE,
                    alpha=0.16, zorder=2, linewidth=0)
    ax.plot(mon["date"], mon["smoke_days"], color=M.SMOKE, linewidth=0.9, zorder=3)
    ax.set_ylabel("smoke days in the month", color=M.INK_2, fontsize=8.5)
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2026-01-01"))
    ax.set_ylim(0, 9.4)
    ax.text(pd.Timestamp("2001-07-01"), 5.6, "Terra\nonly",
            fontsize=7.2, color=M.MUTED, ha="center", va="center",
            linespacing=1.5)

    for date, label, dx, dy in [("2020-09-01", "Sep 2020", -230, 1.3),
                                ("2021-08-01", "Aug 2021", 60, 0.5),
                                ("2023-06-01", "Jun 2023", 150, 1.1)]:
        t = pd.Timestamp(date)
        v = float(mon.loc[mon["date"] == t, "smoke_days"].iloc[0])
        ax.annotate(label, xy=(t, v), xytext=(t + pd.Timedelta(days=dx), v + dy),
                    fontsize=7.0, color=M.INK_2,
                    ha="right" if dx < 0 else "left", va="bottom",
                    arrowprops=dict(arrowstyle="-", linewidth=0.6, color=M.MUTED,
                                    shrinkA=0, shrinkB=2))
    ax.text(0, 1.17, "a   Every month of the archive", transform=ax.transAxes,
            fontsize=9.3, color=M.INK, va="bottom", fontweight="semibold")
    ax.text(0, 1.05,
            "All 306 months, 2000-02 to 2025-07. Shaded: the Terra-only years, "
            "excluded from every trend here.",
            transform=ax.transAxes, fontsize=7.7, color=M.INK_2, va="bottom")

    # --- (b) annual totals ----------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    M.style_axes(ax2)
    early = yr <= EARLY_END
    ax2.bar(yr[early], sd[early], width=0.74, color=PALE, edgecolor="none", zorder=3)
    ax2.bar(yr[~early], sd[~early], width=0.74, color=M.SMOKE, edgecolor="none",
            zorder=3)

    ts = C.theil_sen(yr, sd)
    xx = np.array([yr[0] - 0.6, yr[-1] + 0.6])
    mid = float(np.mean(yr))
    ymid = ts["intercept"] + ts["slope"] * mid
    # The slope uncertainty is a shaded band rather than a pair of dashed
    # lines: the half-means below are the other horizontal reference in this
    # panel, and two dashed families read as one object at print size.
    ax2.fill_between(xx, ymid + ts["slope_lo"] * (xx - mid),
                     ymid + ts["slope_hi"] * (xx - mid),
                     color="#9aa3ac", alpha=0.30, linewidth=0, zorder=5,
                     label="95 % range on the slope")
    ax2.plot(xx, ts["intercept"] + ts["slope"] * xx, color=M.INK, linewidth=1.8,
             zorder=7, label="Theil–Sen fit")

    e_mean, l_mean = sd[early].mean(), sd[~early].mean()
    ax2.hlines(e_mean, 2002.4, EARLY_END + 0.5, color=MEAN_EARLY, linewidth=2.4,
               zorder=6, capstyle="butt",
               label=f"2003–2013 mean, {e_mean:.1f} days")
    ax2.hlines(l_mean, EARLY_END + 0.5, 2024.6, color=MEAN_LATE, linewidth=2.4,
               zorder=6, capstyle="butt",
               label=f"2014–2024 mean, {l_mean:.1f} days")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left",
               bbox_to_anchor=(0.385, 1.025), handlelength=1.9,
               labelspacing=0.42, borderpad=0.2)

    ax2.set_ylabel("smoke days in the year", color=M.INK_2, fontsize=8.5)
    ax2.set_xlim(2002.3, 2024.7)
    ax2.set_ylim(0, 21.8)
    ax2.set_xticks(np.arange(2004, 2025, 4))

    lt = M.log_trend(sd, yr)
    txt = (f"Theil–Sen   {ts['slope']:+.2f} days yr$^{{-1}}$"
           f"   [{ts['slope_lo']:+.2f}, {ts['slope_hi']:+.2f}]\n"
           f"Kendall τ = {ts['kendall_tau']:.2f},   p = {ts['kendall_p']:.3f}\n"
           f"proportional rate   {lt['percent_per_year']:+.1f} % yr$^{{-1}}$")
    ax2.text(0.014, 0.97, txt, transform=ax2.transAxes, fontsize=7.5,
             color=M.INK_2, va="top", ha="left", linespacing=1.75,
             bbox=dict(boxstyle="round,pad=0.42", facecolor=M.SURFACE,
                       edgecolor=M.GRID, linewidth=0.7))

    ax2.text(0, 1.15, "b   Annual totals and the trend", transform=ax2.transAxes,
             fontsize=9.3, color=M.INK, va="bottom", fontweight="semibold")
    ax2.text(0, 1.035,
             "Complete years with both satellites flying. Grey band: the 95 % "
             "range on the slope. Horizontal bars: the mean of each half.",
             transform=ax2.transAxes, fontsize=7.7, color=M.INK_2, va="bottom")

    M.save(fig, "fig2_record.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
