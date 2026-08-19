#!/usr/bin/env python3
"""Concatenate the monthly checkpoints into one archive NetCDF (plan section 2).

Every monthly file shares one CRS, transform and coordinate vector by
construction, so this is a straight concat along `time` -- but the whole point
of freezing the grid is that it can be *checked* rather than assumed, so this
verifies before it combines and refuses rather than silently reindexing.

Also reports what is missing. A gap in the output should be a fact you know
about, not something you discover in a plot two months later: MCD19A2 has real
upstream outages (2002-08-01..07 is a contiguous 7-day hole in CMR itself), and
a month absent from the archive is different from a month present with thin
coverage.

    python3 concat_archive.py data/maiac/monthly -o data/maiac_smoke_25km_monthly.nc
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maiac_pipeline import config, granules, write_netcdf  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("indir", help="directory of maiac_smoke_25km_YYYY_MM.nc files")
    p.add_argument("-o", "--out", default="", help="output NetCDF")
    p.add_argument("--start", default=config.ARCHIVE_START)
    p.add_argument("--end", default="")
    args = p.parse_args(argv)

    import xarray as xr

    paths = sorted(glob.glob(os.path.join(args.indir, "maiac_smoke_25km_*.nc")))
    if not paths:
        print(f"no monthly files in {args.indir}", file=sys.stderr)
        return 1

    found = {os.path.basename(p).split("_25km_")[1][:7].replace("_", "-"): p for p in paths}
    end = args.end or max(found)
    expected = granules.months_in_range(args.start, end)
    missing = [m for m in expected if m not in found]

    print(f"{len(found)} monthly files, {args.start} .. {end}")
    if missing:
        print(f"MISSING {len(missing)} months: {', '.join(missing)}")
    else:
        print("no gaps in the requested range")

    # Verify the frozen grid before combining, rather than letting xarray
    # quietly outer-join two grids that disagree.
    x_ref, y_ref = config.target_coords()
    bad = []
    for month in sorted(found):
        with xr.open_dataset(found[month]) as ds:
            if not (np.allclose(ds["x"].values, x_ref) and np.allclose(ds["y"].values, y_ref)):
                bad.append(month)
    if bad:
        print(f"REFUSING to concat: {len(bad)} months are off-grid: {', '.join(bad)}",
              file=sys.stderr)
        return 1
    print("grid check: all months share the frozen 130 x 242 EPSG:5070 grid")

    ds = xr.open_mfdataset(
        [found[m] for m in sorted(found)],
        combine="nested", concat_dim="time", coords="minimal",
        data_vars="minimal", compat="override",
    ).sortby("time")

    ds.attrs["n_months"] = len(found)
    ds.attrs["time_coverage_start"] = sorted(found)[0]
    ds.attrs["time_coverage_end"] = sorted(found)[-1]
    if missing:
        ds.attrs["missing_months"] = ",".join(missing)

    out = args.out or os.path.join(args.indir, "..", "maiac_smoke_25km_monthly.nc")
    out = os.path.abspath(out)
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4,
               "shuffle": True, "_FillValue": -9999.0}
        for name in write_netcdf.DATA_VARS
    }
    tmp = out + ".tmp"
    ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=encoding)
    os.replace(tmp, out)
    ds.close()

    size_mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({size_mb:.1f} MB, {len(found)} months)")

    with xr.open_dataset(out) as check:
        t = check["time"].values
        print(f"time: {str(t[0])[:7]} .. {str(t[-1])[:7]}, {len(t)} steps")
        print(f"dims: {dict(check.sizes)}")
        w = check["valid_pixel_day_weight"].values
        per_month = np.nansum(w, axis=(1, 2))
        thin = [str(t[i])[:7] for i in np.argsort(per_month)[:5]]
        print(f"five thinnest months by valid pixel-days: {', '.join(thin)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
