#!/usr/bin/env python3
"""Figure 1 -- from MCD19A2 granules to a smoke-day record.

Drawn rather than exported from a diagramming tool, so it regenerates from the
repository like every other figure. Fixed 0-100 coordinate box; reads no data.

    python3 fig1_processing.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import maiac_common as M

X0 = 4.0
W = 66.0
SIDE_X = X0 + W + 5.0
SIDE_W = 25.0

FACE = "#eef4fc"
FACE_OUT = "#e2ecfa"
SIDE_FACE = "#fbf1ea"


def box(ax, x, y, w, h, title, body, edge, face, title_size=8.8, body_size=7.7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
        linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + 1.8, y + h - 2.0, title, fontsize=title_size, color=M.INK,
            va="top", ha="left", fontweight="semibold", zorder=3)
    if body:
        ax.text(x + 1.8, y + h - 5.3, body, fontsize=body_size, color=M.INK_2,
                va="top", ha="left", linespacing=1.55, zorder=3)


def arrow(ax, x, y0, y1, color):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9, color=color,
                                 zorder=1, shrinkA=0, shrinkB=0))


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.3, 8.4), dpi=M.DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 122)
    ax.axis("off")

    ax.text(X0, 120.5, "MODIS MAIAC  ·  MCD19A2.061", fontsize=9.8,
            color=M.SMOKE, fontweight="semibold", va="top")
    ax.text(SIDE_X, 120.5, "reference", fontsize=9.0,
            color=M.HMS_C, fontweight="semibold", va="top")

    steps = [
        ("Granules", 11.5,
         "14 CONUS sinusoidal tiles, 2000-02 → 2025-07, 1,068 GB from NASA.\n"
         "Eight of the 22 tiles a bounding box returns hold no CONUS land."),
        ("Screen every observation, at 1 km", 13.0,
         "land (AOD_QA bits 3–4 = 0)  ·  best quality (bits 8–11 = 0)  ·  AOD ≥ 0\n"
         "the smoke aerosol model is bits 13–14 — a model the retrieval\n"
         "selected, not a measurement of smoke optical depth"),
        ("Collapse to one pixel-day", 13.5,
         "A = mean AOD over the day's valid obs      B = 1 if any valid obs\n"
         "C = mean AOD over smoke-model obs           D = 1 if any smoke obs\n"
         "Terra and Aqua fold into the same day, so a day counts once"),
        ("Mask, then bin to 25 km", 13.0,
         "CONUS mask rasterised at native 1 km in lon/lat, applied before\n"
         "aggregation; counting sort into EPSG:5070 cells; ΣA ΣB ΣC ΣD\n"
         "accumulated in 64-bit over every pixel and every day of the month"),
        ("Monthly archive  ·  306 months, 25 km", 12.5,
         "ΣB  valid pixel-days — the denominator of everything, and the\n"
         "        sample size any ratio must be read against\n"
         "ΣD/ΣB  smoke frequency        ΣC/ΣB  smoke AOD index"),
    ]

    y = 115.0
    for i, (title, h, body) in enumerate(steps):
        y -= h
        box(ax, X0, y, W, h, title, body, M.SMOKE,
            FACE_OUT if i == len(steps) - 1 else FACE)
        if i < len(steps) - 1:
            arrow(ax, X0 + W / 2, y, y - 4.2, M.SMOKE)
            y -= 4.2

    # --- the unit -------------------------------------------------------
    y -= 6.0
    arrow(ax, X0 + W / 2, y + 6.0, y + 1.2, M.SMOKE)
    h = 14.0
    y -= h
    box(ax, X0, y, W, h, "Smoke days  (§2.4)",
        "smoke days in a month = (ΣD / ΣB) × days in the month,\n"
        "summed over the twelve months of a calendar year.\n"
        "One number per 25 km cell per year, in a unit that is not AOD.",
        M.INK, "#f4f3ee")

    # --- outputs --------------------------------------------------------
    y -= 6.0
    arrow(ax, X0 + W / 2, y + 6.0, y + 1.2, M.INK)
    h = 16.5
    y -= h
    box(ax, X0, y, W, h, "Results  (§3–§4)",
        "CONUS record and its trend  ·  frequency versus intensity  ·\n"
        "per-cell trend map and its field significance  ·  seasonal timing  ·\n"
        "how much of the country crosses a given number of smoke days",
        M.INK, M.SURFACE)

    # --- side branch ----------------------------------------------------
    side_h = 30.0
    side_y = 115.0 - side_h
    box(ax, SIDE_X, side_y, SIDE_W, side_h, "NOAA HMS",
        "Analyst-drawn smoke\npolygons, dissolved per\nUTC day, clipped to\nCONUS, binned to a\n0.1° grid.\n\n"
        "Only the heavy density\nclass is used, and only\nin §5.",
        M.HMS_C, SIDE_FACE, title_size=8.4, body_size=7.4)
    # down the margin, then left into the results box
    ax.add_patch(FancyArrowPatch(
        (SIDE_X + SIDE_W / 2, side_y), (X0 + W + 0.4, y + h / 2),
        arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=M.HMS_C,
        zorder=1, linestyle=(0, (3, 2)), shrinkA=2, shrinkB=0,
        connectionstyle="angle,angleA=-90,angleB=180,rad=2"))
    ax.text(SIDE_X + SIDE_W / 2 + 1.4, side_y - 22,
            "independent\ncheck  (§5)", fontsize=7.3, color=M.HMS_C,
            va="center", ha="left", linespacing=1.5)

    fig.tight_layout(pad=0.3)
    M.save(fig, "fig1_processing.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
