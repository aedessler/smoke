"""Read Optical_Depth_055 and AOD_QA out of an MCD19A2 HDF4-EOS granule.

The only module that needs GDAL. Isolated here so everything with scientific
consequences stays testable without an HDF4-capable GDAL build.
"""
from __future__ import annotations

import numpy as np

from . import config


class SubdatasetMissing(RuntimeError):
    """The granule opened, but lacks the 1 km science datasets we need."""


def _find_subdatasets(path: str) -> tuple[str, str]:
    from osgeo import gdal

    container = gdal.Open(path)
    if container is None:
        raise RuntimeError(f"GDAL could not open {path}")
    subs = container.GetSubDatasets()  # [(name, description), ...]
    aod = qa = None
    for name, _desc in subs:
        # Match on the name's suffix rather than the human-readable
        # description: MCD19A2 also carries a grid5km group whose members would
        # otherwise match on a bare substring test.
        if name.endswith(config.SUBDATASET_AOD):
            aod = name
        elif name.endswith(config.SUBDATASET_QA):
            qa = name
    if aod is None or qa is None:
        raise SubdatasetMissing(
            f"{path}: missing grid1km Optical_Depth_055 / AOD_QA "
            f"(found {len(subs)} subdatasets)"
        )
    return aod, qa


def read_granule(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (aod_raw, qa), each shaped (n_observations, 1200, 1200).

    A daily tile holds one band per Terra/Aqua overpass, so the band count
    varies day to day. GDAL collapses a single-band read to 2-D; normalising
    that here means no caller has to remember the special case.
    """
    from osgeo import gdal

    aod_name, qa_name = _find_subdatasets(path)

    aod_ds = gdal.Open(aod_name)
    qa_ds = gdal.Open(qa_name)
    if aod_ds is None or qa_ds is None:
        raise RuntimeError(f"{path}: subdataset opened as None")

    aod_raw = aod_ds.ReadAsArray()
    qa = qa_ds.ReadAsArray()
    if aod_raw is None or qa is None:
        raise RuntimeError(f"{path}: subdataset read returned None")

    if aod_raw.ndim == 2:
        aod_raw = aod_raw[None, :, :]
    if qa.ndim == 2:
        qa = qa[None, :, :]

    if aod_raw.shape != qa.shape:
        raise RuntimeError(
            f"{path}: AOD {aod_raw.shape} and QA {qa.shape} disagree"
        )
    expected = (config.MODIS_TILE_PIXELS, config.MODIS_TILE_PIXELS)
    if aod_raw.shape[1:] != expected:
        raise RuntimeError(f"{path}: unexpected tile shape {aod_raw.shape[1:]}")

    return aod_raw, qa


def describe_granule(path: str) -> dict:
    """Phase A inspection helper: what is actually inside one file."""
    from osgeo import gdal

    aod_name, qa_name = _find_subdatasets(path)
    ds = gdal.Open(aod_name)
    return {
        "path": path,
        "n_observations": ds.RasterCount,
        "shape": (ds.RasterYSize, ds.RasterXSize),
        "projection": ds.GetProjection(),
        "geotransform": ds.GetGeoTransform(),
        "aod_subdataset": aod_name,
        "qa_subdataset": qa_name,
    }
