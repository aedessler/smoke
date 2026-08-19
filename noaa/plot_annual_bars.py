#!/usr/bin/env python3
"""Bar version of the annual smoke-area figure.

Same series, tokens and framing as plot_smoke.py's annual figure, drawn as bars
instead of a line+area, and with no year singled out in the accent color. Reads
the annual table plot_smoke.py already wrote, so it needs no gridded product and
does no aggregation of its own.

    python3 plot_annual_bars.py               # all-density series
    python3 plot_annual_bars.py --class heavy

Writes figures/annual_<cls>_smoke_area_bars.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.use("Agg")

from plot_smoke import (  # noqa: E402  -- reuse the tokens, don't fork them
    AXIS,
    DPI,
    FIG_DIR,
    IN_DIR,
    INK,
    INK_2,
    MUTED,
    SERIES_1,
    SPECS,
    style_axes,
)


def make_annual_bars(spec) -> Path:
    ann = pd.read_csv(IN_DIR / f"annual_{spec.key}_smoke_area.csv").set_index("year")
    y = ann["km2_days"] / 1e6  # million km^2 * days
    med = float(y.median())

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    style_axes(ax)

    ax.bar(y.index, y.values, color=SERIES_1, width=0.72, zorder=3, linewidth=0)

    # Median reference, so the recent years read against a baseline rather than
    # against the eye's guess at the middle of the axis. Drawn over the bars here
    # (a line behind them would vanish), in the recessive axis gray.
    ax.axhline(med, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(f"median {med:,.0f}", xy=(y.index.min() - 0.4, med), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=MUTED)

    ax.set_ylabel("million km² · days", color=INK_2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(y.index[::2])
    ax.margins(x=0.02)
    ax.set_title(
        f"Annual area under {spec.phrase} over the contiguous United States",
        fontsize=13.5, color=INK, pad=30, loc="left",
    )
    ax.text(
        0, 1.025,
        f"Daily {spec.attr} area summed over each year. "
        f"NOAA HMS, {int(y.index.min())}–{int(y.index.max())}.",
        transform=ax.transAxes, fontsize=9.5, color=INK_2,
    )
    fig.tight_layout()
    out = FIG_DIR / f"annual_{spec.key}_smoke_area_bars.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--class", dest="cls", default="all", type=str.lower,
                    choices=sorted(SPECS), help="smoke class to plot (default: all)")
    args = ap.parse_args(argv)
    spec = SPECS[args.cls]

    src = IN_DIR / f"annual_{spec.key}_smoke_area.csv"
    if not src.exists():
        print(f"ERROR: missing {src.name} in data/processed/.\n"
              f"       Run: python3 plot_smoke.py --class {spec.key}", file=sys.stderr)
        return 1

    out = make_annual_bars(spec)
    print(f"  wrote {out.relative_to(Path(__file__).resolve().parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
