#!/usr/bin/env python3
"""End-to-end wiring test with synthetic granules.

Runs a whole month through `process_month` -- day grouping, unit caching,
accumulation, ratio finalisation, NetCDF write and validation -- with the two
things that need the outside world (CMR search, HTTPS download) and the one
thing that needs an HDF4-capable GDAL (granule read) replaced by stubs.

The point is to catch plumbing bugs on a laptop instead of an hour into a spot
VM. The synthetic data is chosen so the correct answer is exact and known:

    every pixel, every day, two observations
      obs 0: AOD 0.100, smoke aerosol model, best quality, land
      obs 1: AOD 0.300, background model,    best quality, land

    => daily_aod 0.2, daily_smoke_aod 0.1, valid_day 1, smoke_day 1
    => mean_aod_055 0.2, mean_smoke_aod_055 0.1,
       smoke_aod_index 0.1, smoke_pixel_day_fraction 1.0
       in every cell that has any CONUS land, regardless of the mask.

    python3 maiac/tests/test_integration.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from maiac_pipeline import (  # noqa: E402
    config, conus_masks, download, granules as gran, hdf_reader, process_month,
)

TILES = ["h09v04", "h10v04", "h10v05"]
DAYS = ["2023-06-15", "2023-06-16"]
DOY = {"2023-06-15": 166, "2023-06-16": 167}


def _fake_granule(name: str) -> dict:
    return {
        "umm": {
            "GranuleUR": name,
            "DataGranule": {
                "ArchiveAndDistributionInformation": [{"Size": 8.0, "SizeUnit": "MB"}]
            },
        }
    }


def _fake_month(month: str, **kwargs) -> list:
    out = []
    for day in DAYS:
        for tile in TILES:
            out.append(
                _fake_granule(f"MCD19A2.A2023{DOY[day]:03d}.{tile}.061.2023200000000")
            )
    return out


def _fake_download(granules, dest, threads=8, attempts=3) -> list[str]:
    os.makedirs(dest, exist_ok=True)
    return [os.path.join(dest, gran.granule_name(g) + ".hdf") for g in granules]


def _fake_read(path: str):
    n = config.MODIS_TILE_PIXELS
    aod = np.empty((2, n, n), dtype="int16")
    aod[0], aod[1] = 100, 300
    qa = np.empty((2, n, n), dtype="uint16")
    qa[0] = 1 << 13  # land, best quality, smoke model
    qa[1] = 0        # land, best quality, background model
    return aod, qa


class TestMonthEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = (
            gran.search_month, download.download_granules, hdf_reader.read_granule
        )
        gran.search_month = _fake_month
        process_month.gran.search_month = _fake_month
        download.download_granules = _fake_download
        hdf_reader.read_granule = _fake_read

        cls.tmp = tempfile.mkdtemp(prefix="maiac_it_")
        cls.out = os.path.join(cls.tmp, "output")
        cls.scratch = os.path.join(cls.tmp, "raw")
        cls.cache = os.path.join(cls.tmp, "cache")
        conus_masks.build_all_masks(TILES, cls.cache)

        cls.result = process_month.process_month(
            "2023-06", outdir=cls.out, scratch=cls.scratch, cache_dir=cls.cache,
            bucket_uri=None, tiles=tuple(TILES),
        )

    @classmethod
    def tearDownClass(cls):
        gran.search_month, download.download_granules, hdf_reader.read_granule = cls._saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_month_reported_success(self):
        self.assertNotIn("FAILED", self.result, self.result)

    def test_one_unit_cached_per_day(self):
        units = sorted(os.listdir(os.path.join(self.out, "units", "2023-06")))
        self.assertEqual(units, [f"{d}.npz" for d in DAYS])

    def test_no_tmp_files_survive(self):
        leftovers = [
            os.path.join(root, f)
            for root, _d, files in os.walk(self.out)
            for f in files if ".tmp" in f
        ]
        self.assertEqual(leftovers, [])

    def test_raw_scratch_deleted(self):
        month_scratch = os.path.join(self.scratch, "2023-06")
        self.assertFalse(os.path.exists(month_scratch), "raw month was not cleaned up")

    def test_netcdf_written_and_valid(self):
        import xarray as xr

        path = os.path.join(self.out, "monthly", "maiac_smoke_25km_2023_06.nc")
        self.assertTrue(os.path.exists(path))
        with xr.open_dataset(path) as ds:
            from maiac_pipeline import write_netcdf

            self.assertEqual(write_netcdf.validate(ds), [])
            self.assertEqual(ds.sizes["y"], config.TARGET_NY)
            self.assertEqual(ds.sizes["x"], config.TARGET_NX)
            self.assertEqual(ds.sizes["time"], 1)
            self.assertEqual(
                str(ds["time"].values[0])[:10], "2023-06-01",
                "time must label the first day of the month",
            )

    def test_values_match_the_injected_truth(self):
        import xarray as xr

        path = os.path.join(self.out, "monthly", "maiac_smoke_25km_2023_06.nc")
        with xr.open_dataset(path) as ds:
            covered = ds["valid_pixel_day_weight"].values[0] > 0
            self.assertGreater(covered.sum(), 1000, "suspiciously little coverage")

            def cov(name):
                return ds[name].values[0][covered]

            np.testing.assert_allclose(cov("mean_aod_055"), 0.2, rtol=1e-4)
            np.testing.assert_allclose(cov("mean_smoke_aod_055"), 0.1, rtol=1e-4)
            np.testing.assert_allclose(cov("smoke_aod_index"), 0.1, rtol=1e-4)
            np.testing.assert_allclose(cov("smoke_pixel_day_fraction"), 1.0, rtol=1e-4)

            # Two days of observations, so the weight is 2x the native pixel
            # count binned into each cell -- and never below it.
            weight = cov("valid_pixel_day_weight")
            self.assertTrue(np.all(weight >= 2))
            np.testing.assert_allclose(
                cov("smoke_pixel_day_weight"), weight, rtol=1e-5
            )

    def test_uncovered_cells_are_nan_not_zero(self):
        import xarray as xr

        path = os.path.join(self.out, "monthly", "maiac_smoke_25km_2023_06.nc")
        with xr.open_dataset(path) as ds:
            empty = ds["valid_pixel_day_weight"].values[0] == 0
            self.assertTrue(empty.any(), "expected some cells outside the tiles used")
            self.assertTrue(np.all(np.isnan(ds["mean_aod_055"].values[0][empty])))

    def test_manifest_record_written(self):
        from maiac_pipeline import manifest

        rec = manifest.read_record(self.out, "2023-06")
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["n_days"], len(DAYS))
        self.assertEqual(rec["n_granules"], len(DAYS) * len(TILES))
        self.assertEqual(rec["n_days_partial"], 0)

    def test_rerun_is_idempotent_and_skips(self):
        again = process_month.process_month(
            "2023-06", outdir=self.out, scratch=self.scratch, cache_dir=self.cache,
            bucket_uri=None, tiles=tuple(TILES),
        )
        self.assertIn("already complete", again)


class TestUnitResume(unittest.TestCase):
    """A month whose NetCDF was lost must rebuild from cached units, not
    re-download -- that is the whole point of the unit cache surviving a
    preemption."""

    def test_rebuild_uses_cached_units(self):
        saved = (gran.search_month, download.download_granules, hdf_reader.read_granule)
        gran.search_month = _fake_month
        process_month.gran.search_month = _fake_month
        download.download_granules = _fake_download
        hdf_reader.read_granule = _fake_read
        tmp = tempfile.mkdtemp(prefix="maiac_resume_")
        try:
            out = os.path.join(tmp, "output")
            cache = os.path.join(tmp, "cache")
            conus_masks.build_all_masks(TILES, cache)
            kwargs = dict(
                outdir=out, scratch=os.path.join(tmp, "raw"), cache_dir=cache,
                bucket_uri=None, tiles=tuple(TILES),
            )
            process_month.process_month("2023-06", **kwargs)

            # Simulate losing the deliverable but keeping the resumable cache.
            os.remove(os.path.join(out, "monthly", "maiac_smoke_25km_2023_06.nc"))

            calls = []

            def _explode(granules, dest, threads=8, attempts=3):
                calls.append(dest)
                raise AssertionError("re-downloaded a day that was already cached")

            download.download_granules = _explode
            result = process_month.process_month("2023-06", **kwargs)
            self.assertNotIn("FAILED", result, result)
            self.assertEqual(calls, [], "no download should have been attempted")
        finally:
            gran.search_month, download.download_granules, hdf_reader.read_granule = saved
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
