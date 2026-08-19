#!/usr/bin/env python3
"""Four-panel spot check for one monthly MAIAC file (plan section 24).

June 2023 is the month to look at first: the Canadian wildfire smoke episode
put a large, unambiguous signal over the eastern U.S., so a pipeline that is
wrong about smoke will look obviously wrong rather than subtly wrong.

The grid is already in EPSG:5070, a projected CRS, so a plain imshow with the
correct extent IS a correct map -- no cartopy needed. The state outline is
reprojected to match.

    python3 plot_month.py data/maiac/monthly/maiac_smoke_25km_2023_06.nc
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maiac_pipeline import config  # noqa: E402

PANELS = [
    ("mean_aod_055", "Mean AOD (550 nm)", "viridis", None),
    ("mean_smoke_aod_055", "Mean AOD | smoke model", "inferno", None),
    ("smoke_aod_index", "Smoke AOD index\nsum(smoke AOD) / sum(valid pixel-days)", "magma", None),
    ("smoke_pixel_day_fraction", "Smoke pixel-day fraction", "cividis", (0, 1)),
]


def _states(ax_crs="EPSG:5070"):
    """CONUS state outlines in the target CRS, or None if unavailable."""
    try:
        import geopandas as gpd

        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".plotcache")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, "states.geojson")
        if not os.path.exists(path):
            import requests

            zip_path = os.path.join(cache, "states.zip")
            if not os.path.exists(zip_path):
                r = requests.get(config.CENSUS_STATES_URL, timeout=180)
                r.raise_for_status()
                with open(zip_path, "wb") as fh:
                    fh.write(r.content)
            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                shp = next(n for n in zf.namelist() if n.endswith(".shp"))
            gdf = gpd.read_file(f"zip://{zip_path}!{shp}")
            gdf = gdf[~gdf["STATEFP"].isin(config.NON_CONUS_FIPS)]
            gdf.to_file(path, driver="GeoJSON")
        return gpd.read_file(path).to_crs(ax_crs)
    except Exception as exc:  # outlines are decoration, not the point
        print(f"  (state outlines unavailable: {exc})")
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("netcdf")
    p.add_argument("-o", "--out", default="", help="output PNG (default alongside input)")
    p.add_argument("--min-weight", type=float, default=5.0,
                   help="hide cells with fewer valid pixel-days than this")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    ds = xr.open_dataset(args.netcdf)
    month = str(ds["time"].values[0])[:7]
    weight = ds["valid_pixel_day_weight"].values[0]

    # A cell with three valid pixel-days produces a mean that looks exactly as
    # authoritative as one with thirty thousand. Hide the former.
    thin = weight < args.min_weight
    extent = [config.TARGET_XMIN, config.TARGET_XMAX, config.TARGET_YMIN, config.TARGET_YMAX]
    states = _states()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (name, title, cmap, lim) in zip(axes.ravel(), PANELS):
        data = np.where(thin, np.nan, ds[name].values[0])
        vmin, vmax = lim if lim else (
            np.nanpercentile(data, 2) if np.isfinite(data).any() else 0,
            np.nanpercentile(data, 98) if np.isfinite(data).any() else 1,
        )
        im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        if states is not None:
            states.boundary.plot(ax=ax, linewidth=0.4, edgecolor="white", alpha=0.55)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.01)

    covered = int(np.sum(weight > 0))
    fig.suptitle(
        f"MAIAC MCD19A2.061 — {month} — 25 km CONUS (EPSG:5070)\n"
        f"{covered:,} cells with data · median valid pixel-days/cell "
        f"{np.median(weight[weight > 0]):,.0f} · cells below {args.min_weight:g} hidden",
        fontsize=12,
    )

    out = args.out or os.path.splitext(args.netcdf)[0] + ".png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print("\n--- diagnostics ---")
    from maiac_pipeline import write_netcdf

    problems = write_netcdf.validate(ds)
    print("validation:", "clean" if not problems else "; ".join(problems))
    for k, v in write_netcdf.diagnostics(ds).items():
        print(f"  {k}: {v}")
    ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
