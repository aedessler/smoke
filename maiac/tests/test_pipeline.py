#!/usr/bin/env python3
"""Unit tests for the scientific transformations.

Deliberately network-free and GDAL-free: everything with a scientific
consequence (QA bit positions, the pixel-day collapse, duplicate resolution,
the frozen grid, the ratio-of-sums) is tested without needing credentials or an
HDF4-capable GDAL build, so these run on a laptop as well as on the VM.

    python3 -m unittest discover -s maiac/tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from maiac_pipeline import config, granules, qa, regrid  # noqa: E402


class TestFilenameParser(unittest.TestCase):
    NAME = "MCD19A2.A2022032.h13v12.061.2024226235541.hdf"

    def test_parses_every_field(self):
        gid = granules.parse_granule_name(self.NAME)
        self.assertEqual(gid.year, 2022)
        self.assertEqual(gid.doy, 32)
        self.assertEqual(gid.h, 13)
        self.assertEqual(gid.v, 12)
        self.assertEqual(gid.collection, "061")
        self.assertEqual(gid.production, 2024226235541)
        self.assertEqual(gid.tile, "h13v12")

    def test_doy_to_calendar_date(self):
        self.assertEqual(granules.parse_granule_name(self.NAME).date.isoformat(), "2022-02-01")
        # Leap-year boundary: 2020 doy 060 is 29 Feb, not 1 Mar.
        gid = granules.parse_granule_name("MCD19A2.A2020060.h09v04.061.2020100000000.hdf")
        self.assertEqual(gid.date.isoformat(), "2020-02-29")

    def test_accepts_ur_and_url(self):
        ur = "MCD19A2.A2023166.h09v03.061.2023167172304"
        url = "https://data.lpdaac.earthdatacloud.nasa.gov/x/" + ur + ".hdf"
        self.assertEqual(granules.parse_granule_name(ur).h, 9)
        self.assertEqual(granules.parse_granule_name(url).h, 9)

    def test_rejects_other_products(self):
        with self.assertRaises(ValueError):
            granules.parse_granule_name("MOD04_L2.A2023166.1200.061.hdf")


class TestDuplicateSelection(unittest.TestCase):
    def test_keeps_newest_production_timestamp(self):
        older = "MCD19A2.A2022032.h13v12.061.2022100000000.hdf"
        newer = "MCD19A2.A2022032.h13v12.061.2024226235541.hdf"
        self.assertEqual(granules.keep_latest_reprocessing([older, newer]), [newer])
        # Order of arrival must not change the winner.
        self.assertEqual(granules.keep_latest_reprocessing([newer, older]), [newer])

    def test_different_tiles_and_days_are_not_duplicates(self):
        names = [
            "MCD19A2.A2022032.h13v12.061.2022100000000.hdf",
            "MCD19A2.A2022032.h13v11.061.2022100000000.hdf",
            "MCD19A2.A2022033.h13v12.061.2022100000000.hdf",
        ]
        self.assertEqual(len(granules.keep_latest_reprocessing(names)), 3)

    def test_preserves_input_order(self):
        names = [
            "MCD19A2.A2022033.h13v12.061.2022100000000.hdf",
            "MCD19A2.A2022032.h13v12.061.2022100000000.hdf",
        ]
        self.assertEqual(granules.keep_latest_reprocessing(names), names)


class TestMonthHelpers(unittest.TestCase):
    def test_month_bounds_handles_february(self):
        self.assertEqual(granules.month_bounds("2023-02"), ("2023-02-01", "2023-02-28"))
        self.assertEqual(granules.month_bounds("2024-02"), ("2024-02-01", "2024-02-29"))
        self.assertEqual(granules.month_bounds("2023-06"), ("2023-06-01", "2023-06-30"))

    def test_months_in_range_crosses_year_boundary(self):
        self.assertEqual(
            granules.months_in_range("2022-11", "2023-02"),
            ["2022-11", "2022-12", "2023-01", "2023-02"],
        )
        self.assertEqual(granules.months_in_range("2023-06", "2023-06"), ["2023-06"])

    def test_full_archive_length(self):
        months = granules.months_in_range("2000-02", "2025-07")
        self.assertEqual(months[0], "2000-02")
        self.assertEqual(months[-1], "2025-07")
        self.assertEqual(len(months), 306)


class TestQABits(unittest.TestCase):
    """Bit positions are the single easiest thing to get silently wrong."""

    def test_field_extraction(self):
        # surface=2 (snow) at bits 3-4, quality=11 at bits 8-11, model=2 (dust)
        value = (2 << 3) | (11 << 8) | (2 << 13)
        fields = qa.decode_qa(np.array([[value]], dtype="uint16"))
        self.assertEqual(fields["surface"][0, 0], 2)
        self.assertEqual(fields["quality"][0, 0], 11)
        self.assertEqual(fields["model"][0, 0], 2)

    def test_land_best_quality_smoke_is_valid_and_smoke(self):
        qa_smoke = np.array([[1 << 13]], dtype="uint16")  # land, quality 0, model 1
        aod = np.array([[150]], dtype="int16")
        valid, smoke = qa.valid_masks(aod, qa_smoke)
        self.assertTrue(valid[0, 0])
        self.assertTrue(smoke[0, 0])

    def test_background_model_is_valid_but_not_smoke(self):
        valid, smoke = qa.valid_masks(
            np.array([[150]], dtype="int16"), np.array([[0]], dtype="uint16")
        )
        self.assertTrue(valid[0, 0])
        self.assertFalse(smoke[0, 0])

    def test_water_surface_is_rejected(self):
        valid, _ = qa.valid_masks(
            np.array([[150]], dtype="int16"), np.array([[1 << 3]], dtype="uint16")
        )
        self.assertFalse(valid[0, 0])

    def test_fill_value_is_rejected(self):
        valid, _ = qa.valid_masks(
            np.array([[-28672]], dtype="int16"), np.array([[0]], dtype="uint16")
        )
        self.assertFalse(valid[0, 0])

    def test_quality_11_rejected_by_primary_accepted_by_permissive(self):
        aod = np.array([[150]], dtype="int16")
        qa_11 = np.array([[11 << 8]], dtype="uint16")
        self.assertFalse(qa.valid_masks(aod, qa_11, config.QUALITY_PRIMARY)[0][0, 0])
        self.assertTrue(qa.valid_masks(aod, qa_11, config.QUALITY_PERMISSIVE)[0][0, 0])

    def test_dust_model_is_not_smoke(self):
        _, smoke = qa.valid_masks(
            np.array([[150]], dtype="int16"), np.array([[2 << 13]], dtype="uint16")
        )
        self.assertFalse(smoke[0, 0])


class TestPixelDayCollapse(unittest.TestCase):
    def test_averages_over_valid_observations_only(self):
        # Three overpasses on one pixel: smoke, background, fill.
        aod = np.array([[[100]], [[200]], [[-28672]]], dtype="int16")
        qa_arr = np.array([[[1 << 13]], [[0]], [[0]]], dtype="uint16")
        out = qa.collapse_to_pixel_day(aod, qa_arr)
        self.assertAlmostEqual(out["A"][0, 0], 0.15, places=6)  # (0.1 + 0.2) / 2
        self.assertEqual(out["B"][0, 0], 1.0)
        self.assertAlmostEqual(out["C"][0, 0], 0.10, places=6)  # smoke obs only
        self.assertEqual(out["D"][0, 0], 1.0)

    def test_no_valid_observation_yields_all_zero(self):
        aod = np.array([[[-28672]], [[-28672]]], dtype="int16")
        qa_arr = np.zeros((2, 1, 1), dtype="uint16")
        out = qa.collapse_to_pixel_day(aod, qa_arr)
        for key in "ABCD":
            self.assertEqual(out[key][0, 0], 0.0, msg=key)

    def test_valid_but_no_smoke_zeroes_only_the_smoke_fields(self):
        out = qa.collapse_to_pixel_day(
            np.array([[[300]]], dtype="int16"), np.array([[[0]]], dtype="uint16")
        )
        self.assertAlmostEqual(out["A"][0, 0], 0.3, places=6)
        self.assertEqual(out["B"][0, 0], 1.0)
        self.assertEqual(out["C"][0, 0], 0.0)
        self.assertEqual(out["D"][0, 0], 0.0)

    def test_single_observation_2d_input_is_normalised(self):
        """GDAL returns 2-D for a one-overpass day; reducing the wrong axis here
        would collapse the image, not the observations."""
        out2d = qa.collapse_to_pixel_day(
            np.full((4, 5), 250, dtype="int16"), np.zeros((4, 5), dtype="uint16")
        )
        self.assertEqual(out2d["A"].shape, (4, 5))
        np.testing.assert_allclose(out2d["A"], 0.25, rtol=1e-6)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            qa.collapse_to_pixel_day(
                np.zeros((2, 4, 4), dtype="int16"), np.zeros((3, 4, 4), dtype="uint16")
            )


class TestTargetGrid(unittest.TestCase):
    def test_frozen_dimensions(self):
        self.assertEqual(config.TARGET_NX, 242)
        self.assertEqual(config.TARGET_NY, 130)
        self.assertEqual(config.TARGET_NCELLS, 31460)

    def test_coordinates_are_cell_centres_inside_the_bounds(self):
        x, y = config.target_coords()
        self.assertEqual(x[0], config.TARGET_XMIN + config.TARGET_RES / 2)
        self.assertEqual(y[0], config.TARGET_YMAX - config.TARGET_RES / 2)
        self.assertTrue(np.all(x > config.TARGET_XMIN) and np.all(x < config.TARGET_XMAX))
        self.assertTrue(np.all(y > config.TARGET_YMIN) and np.all(y < config.TARGET_YMAX))

    def test_y_descends_north_up(self):
        _, y = config.target_coords()
        self.assertTrue(np.all(np.diff(y) < 0))

    def test_tile_pixel_centres_land_inside_the_tile(self):
        x, y = config.tile_pixel_centres(9, 4)
        self.assertEqual(x.shape, (1200, 1200))
        x0 = config.MODIS_X_MIN + 9 * config.MODIS_TILE_SIZE
        y0 = config.MODIS_Y_MAX - 4 * config.MODIS_TILE_SIZE
        self.assertGreater(x.min(), x0)
        self.assertLess(x.max(), x0 + config.MODIS_TILE_SIZE)
        self.assertLess(y.max(), y0)
        self.assertGreater(y.min(), y0 - config.MODIS_TILE_SIZE)

    def test_tile_h09v04_is_over_the_western_united_states(self):
        """Catches a sign or tile-origin error in the sinusoidal geometry."""
        from pyproj import Transformer

        x, y = config.tile_pixel_centres(9, 4)
        lon, lat = Transformer.from_crs(
            config.MODIS_SINU_PROJ, "EPSG:4326", always_xy=True
        ).transform(x[600, 600], y[600, 600])
        self.assertTrue(-125 < lon < -100, f"lon {lon}")
        self.assertTrue(35 < lat < 50, f"lat {lat}")


class TestRegrid(unittest.TestCase):
    def test_accumulate_bins_pixels_into_the_right_cells(self):
        acc = regrid.empty_accumulator()
        cell_index = np.array([[0, 1], [0, -1]], dtype="int32")
        fields = {
            "A": np.array([[0.2, 0.5], [0.4, 9.9]]),
            "B": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "C": np.zeros((2, 2)),
            "D": np.zeros((2, 2)),
        }
        regrid.accumulate_tile(acc, cell_index, fields)
        self.assertAlmostEqual(acc["A"][0], 0.6)   # 0.2 + 0.4
        self.assertAlmostEqual(acc["A"][1], 0.5)
        self.assertAlmostEqual(acc["B"][0], 2.0)
        self.assertAlmostEqual(acc["B"][1], 1.0)
        # The -1 pixel (value 9.9) must not appear anywhere.
        self.assertAlmostEqual(acc["A"].sum(), 1.1)

    def test_fully_masked_tile_is_a_no_op(self):
        acc = regrid.empty_accumulator()
        regrid.accumulate_tile(
            acc,
            np.full((2, 2), -1, dtype="int32"),
            {k: np.ones((2, 2)) for k in regrid.FIELDS},
        )
        self.assertEqual(acc["A"].sum(), 0.0)


class TestMonthlyRatios(unittest.TestCase):
    def _acc(self, a, b, c, d):
        acc = regrid.empty_accumulator()
        for key, val in zip(regrid.FIELDS, (a, b, c, d)):
            acc[key][0] = val
        return acc

    def test_ratio_of_sums(self):
        # 10 valid pixel-days summing to AOD 2.0; 4 of them smoke summing to 1.2
        out = regrid.monthly_ratios(self._acc(2.0, 10.0, 1.2, 4.0))
        self.assertAlmostEqual(out["mean_aod_055"][0, 0], 0.20)
        self.assertAlmostEqual(out["mean_smoke_aod_055"][0, 0], 0.30)
        self.assertAlmostEqual(out["smoke_aod_index"][0, 0], 0.12)
        self.assertAlmostEqual(out["smoke_pixel_day_fraction"][0, 0], 0.40)
        self.assertAlmostEqual(out["valid_pixel_day_weight"][0, 0], 10.0)
        self.assertAlmostEqual(out["smoke_pixel_day_weight"][0, 0], 4.0)

    def test_zero_denominator_is_nan_not_zero(self):
        """'No retrieval all month' and 'retrievals averaging zero' are
        different statements and must not collapse to the same number."""
        out = regrid.monthly_ratios(self._acc(0.0, 0.0, 0.0, 0.0))
        self.assertTrue(np.isnan(out["mean_aod_055"][0, 0]))
        self.assertTrue(np.isnan(out["smoke_pixel_day_fraction"][0, 0]))
        self.assertEqual(out["valid_pixel_day_weight"][0, 0], 0.0)

    def test_valid_but_no_smoke_gives_zero_index_and_nan_conditional_mean(self):
        out = regrid.monthly_ratios(self._acc(2.0, 10.0, 0.0, 0.0))
        self.assertAlmostEqual(out["smoke_aod_index"][0, 0], 0.0)
        self.assertAlmostEqual(out["smoke_pixel_day_fraction"][0, 0], 0.0)
        self.assertTrue(np.isnan(out["mean_smoke_aod_055"][0, 0]))

    def test_ratio_of_sums_differs_from_mean_of_means(self):
        """The reason section 15 exists: a cell with uneven coverage must be
        weighted by how much valid data each part contributed."""
        # Cell A: 30 pixel-days of AOD 0.1. Cell B: 1 pixel-day of AOD 0.9.
        acc = regrid.empty_accumulator()
        acc["A"][0], acc["B"][0] = 3.0, 30.0
        acc["A"][1], acc["B"][1] = 0.9, 1.0
        out = regrid.monthly_ratios(acc)
        flat_aod = out["mean_aod_055"].ravel()
        self.assertAlmostEqual(flat_aod[0], 0.1)
        self.assertAlmostEqual(flat_aod[1], 0.9)
        # Pooling the two cells: ratio of sums = 3.9/31 = 0.1258,
        # while the mean of the two cell means would be 0.5.
        pooled = (acc["A"][0] + acc["A"][1]) / (acc["B"][0] + acc["B"][1])
        self.assertAlmostEqual(pooled, 3.9 / 31.0)
        self.assertNotAlmostEqual(pooled, 0.5, places=2)

    def test_shape_is_the_frozen_grid(self):
        out = regrid.monthly_ratios(regrid.empty_accumulator())
        for name, arr in out.items():
            self.assertEqual(arr.shape, (config.TARGET_NY, config.TARGET_NX), msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
