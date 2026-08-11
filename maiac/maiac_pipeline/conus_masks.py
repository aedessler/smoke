"""Per-tile native-resolution CONUS mask fused with the 25 km cell index.

Two jobs that the plan keeps separate (sections 14 and 17) are done in one
cached array here, because they answer the same question about the same pixel:

    for each of the 1200 x 1200 native pixels in MODIS tile hHHvVV,
    which flat 25 km EPSG:5070 cell does it fall in, or -1 if it should be
    dropped (outside CONUS, or outside the target grid)?

Caching that as one int32 array per tile means the expensive geometry --
sinusoidal -> lon/lat -> point-in-polygon, and sinusoidal -> EPSG:5070 -> cell
index -- happens 22 times for the whole archive rather than once per tile-day.

DELIBERATE DEVIATION from plan section 17. The plan reaches for
`gdalwarp -r sum` on a four-band temporary raster per tile-day. This module
bins each native pixel into the 25 km cell containing its centre and sums with
`np.bincount` instead. The property the plan actually requires is preserved
exactly -- A, B, C and D go through *identical* spatial weighting, so their
ratios are still coverage-weighted 25 km means -- and binning is both exact
(no resampling kernel, every observation counted once and only once) and far
cheaper: it removes ~660 gdalwarp invocations per month. At a 27:1 linear
downsample ratio the difference from area-weighted partial-pixel overlap is
confined to cell edges and is sub-percent.

Point-in-polygon is evaluated in lon/lat, NOT in EPSG:5070. Reprojecting the
CONUS outline into Albers first would chord its long straight segments -- the
49th-parallel border is a single straight line in lon/lat and a curve in
Albers -- which would misplace the northern border by tens of km.
"""
from __future__ import annotations

import logging
import os
import zipfile

import numpy as np

from . import config

log = logging.getLogger("maiac.masks")


def conus_geometry(cache_dir: str):
    """Dissolved CONUS land polygon in lon/lat, cached as GeoJSON."""
    import geopandas as gpd

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "conus.geojson")
    if os.path.exists(path):
        return gpd.read_file(path).geometry.iloc[0]

    import requests

    log.info("downloading CONUS state boundaries")
    zip_path = os.path.join(cache_dir, "cb_us_state_20m.zip")
    if not os.path.exists(zip_path):
        resp = requests.get(config.CENSUS_STATES_URL, timeout=180)
        resp.raise_for_status()
        tmp = zip_path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(resp.content)
        os.replace(tmp, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        shp = next(n for n in zf.namelist() if n.endswith(".shp"))
    states = gpd.read_file(f"zip://{zip_path}!{shp}")

    states = states[~states["STATEFP"].isin(config.NON_CONUS_FIPS)]
    if len(states) != 49:  # 48 states + DC
        raise RuntimeError(f"expected 49 CONUS features, got {len(states)}")
    geom = states.to_crs("EPSG:4326").geometry.union_all()

    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return geom


def tile_cell_index(h: int, v: int, cache_dir: str) -> np.ndarray:
    """(1200, 1200) int32 of flat 25 km cell indices; -1 means drop.

    Cached to `cache_dir/conus_tile_masks/hHHvVV.npy`.
    """
    import shapely

    tile = f"h{h:02d}v{v:02d}"
    mask_dir = os.path.join(cache_dir, "conus_tile_masks")
    os.makedirs(mask_dir, exist_ok=True)
    path = os.path.join(mask_dir, f"{tile}.npy")
    if os.path.exists(path):
        return np.load(path)

    from pyproj import Transformer

    log.info("building cell index for tile %s", tile)
    sx, sy = config.tile_pixel_centres(h, v)

    to_albers = Transformer.from_crs(
        config.MODIS_SINU_PROJ, config.TARGET_CRS, always_xy=True
    )
    ax, ay = to_albers.transform(sx, sy)

    col = np.floor((ax - config.TARGET_XMIN) / config.TARGET_RES)
    row = np.floor((config.TARGET_YMAX - ay) / config.TARGET_RES)
    inside = (
        np.isfinite(ax)
        & np.isfinite(ay)
        & (col >= 0)
        & (col < config.TARGET_NX)
        & (row >= 0)
        & (row < config.TARGET_NY)
    )

    # Only pixels that landed on the grid are worth a point-in-polygon test.
    if inside.any():
        to_lonlat = Transformer.from_crs(
            config.MODIS_SINU_PROJ, "EPSG:4326", always_xy=True
        )
        lon, lat = to_lonlat.transform(sx[inside], sy[inside])
        geom = conus_geometry(cache_dir)
        shapely.prepare(geom)
        in_conus = shapely.contains_xy(geom, lon, lat)
        inside[inside] = in_conus

    idx = np.full(sx.shape, -1, dtype="int32")
    idx[inside] = (row[inside] * config.TARGET_NX + col[inside]).astype("int32")

    # ".tmp.npy", not ".tmp": np.save silently appends ".npy" to any name that
    # lacks it, which would write the array somewhere the rename can't find it.
    tmp = path + ".tmp.npy"
    np.save(tmp, idx)
    os.replace(tmp, path)
    log.info("tile %s: %d/%d pixels retained", tile, int(inside.sum()), inside.size)
    return idx


def build_all_masks(tiles: list[str], cache_dir: str) -> dict[str, int]:
    """Warm the cache for every tile up front, single-threaded.

    Called once before the process pool starts so that N workers do not race to
    build the same tile mask N times.

    Returns {tile: retained_pixel_count}, which is what lets the caller drop
    tiles that contain no CONUS land at all -- see `useful_tiles`.
    """
    counts = {}
    for tile in sorted(set(tiles)):
        h, v = int(tile[1:3]), int(tile[4:6])
        counts[tile] = int((tile_cell_index(h, v, cache_dir) >= 0).sum())
    return counts


def useful_tiles(tiles: list[str], cache_dir: str) -> list[str]:
    """The subset of `tiles` that actually overlaps CONUS land.

    The CMR bounding-box query is deliberately generous, so it returns tiles
    that merely intersect the box: the whole v03 row is 50-60 N (north of the
    49th parallel), h07v05/h07v06 are Pacific, h11v06 is Atlantic. Eight of the
    22 contribute literally nothing. Not downloading them is ~36% less
    transfer for an identical result -- worth having, given cross-cloud HTTPS
    is the pipeline's bottleneck.
    """
    counts = build_all_masks(tiles, cache_dir)
    keep = [t for t in tiles if counts.get(t, 0) > 0]
    dropped = [t for t in tiles if counts.get(t, 0) == 0]
    if dropped:
        log.info("skipping %d tiles with no CONUS land: %s", len(dropped), ", ".join(dropped))
    return keep
