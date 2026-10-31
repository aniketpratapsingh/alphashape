#!/usr/bin/env python3
"""Unit tests for the bench metrics, on synthetic data only.

Every assertion here has an answer known by construction: a ramp whose
resolution is exact, a mask with a counted number of holes, a shift applied by
hand, a camera whose projection is checked against a hand-computed pixel. That is
what lets the metric code be trusted in a weekly job where the ground truth is
itself another pipeline's output.

Run with plain unittest (no pytest), from the repository root:

    python3 -m unittest discover -s tools/bench/tests -v
"""

import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics  # noqa: E402


def grid(width, height, *, origin=(0.0, 0.0), pixel=(1.0, 1.0)):
    """A Raster with a synthetic payload supplied by the caller."""
    return metrics.Raster(
        data=np.zeros((height, width), dtype=np.float64),
        origin_x=origin[0],
        origin_y=origin[1],
        pixel_size_x=pixel[0],
        pixel_size_y=pixel[1],
    )


class RasterTest(unittest.TestCase):
    def test_extent_uses_the_outer_corner(self):
        raster = metrics.Raster(
            data=np.zeros((10, 20)),
            origin_x=100.0,
            origin_y=200.0,
            pixel_size_x=0.5,
            pixel_size_y=0.25,
        )
        self.assertEqual(raster.extent(), (100.0, 200.0 - 10 * 0.25, 100.0 + 20 * 0.5, 200.0))

    def test_validity_honours_nodata(self):
        data = np.array([[1.0, -32767.0], [np.nan, 4.0]])
        raster = metrics.Raster(data, 0.0, 0.0, 1.0, 1.0, nodata=-32767.0)
        self.assertEqual(raster.validity().tolist(), [[True, False], [False, True]])

    def test_validity_survives_float32_rounding_of_the_sentinel(self):
        # A float32 GeoTIFF cannot hold -32767.0 exactly; the stored value is
        # -32767.0 rounded to float32, which is a hair off. The tolerance must
        # absorb that, or every nodata cell would count as a measurement.
        sentinel = np.float32(-32767.0).astype(np.float64)
        data = np.array([[sentinel, 3.0]])
        raster = metrics.Raster(data, 0.0, 0.0, 1.0, 1.0, nodata=-32767.0)
        self.assertEqual(raster.validity().tolist(), [[False, True]])

    def test_validity_without_nodata_is_finiteness(self):
        raster = metrics.Raster(np.array([[1.0, np.nan], [0.0, -1e9]]), 0, 0, 1, 1)
        self.assertEqual(raster.validity().tolist(), [[True, False], [True, True]])

    def test_bands_and_band_accessor(self):
        raster = metrics.Raster(np.zeros((4, 5, 3)), 0, 0, 1, 1)
        self.assertEqual(raster.bands, 3)
        self.assertEqual(raster.band(2).shape, (4, 5))

    def test_single_band_rejects_a_second_band(self):
        raster = metrics.Raster(np.zeros((2, 2)), 0, 0, 1, 1)
        self.assertEqual(raster.bands, 1)
        with self.assertRaises(IndexError):
            raster.band(1)

    def test_rejects_a_non_positive_pixel_size(self):
        with self.assertRaises(ValueError):
            metrics.Raster(np.zeros((2, 2)), 0, 0, 0.0, 1.0)


class ResampleTest(unittest.TestCase):
    def test_identity_resample_reproduces_the_source(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        source = metrics.Raster(data, 0.0, 3.0, 1.0, 1.0)
        values, valid = metrics.resample_to_grid(
            source, origin_x=0.0, origin_y=3.0, pixel_size=1.0, width=4, height=3
        )
        np.testing.assert_allclose(values[:, :, 0], data)
        self.assertTrue(valid.all())

    def test_bilinear_recovers_a_linear_ramp_exactly(self):
        # A linear ramp is reproduced exactly by bilinear interpolation at any
        # half-pixel offset, which makes it the right probe for a coordinate bug.
        rows = np.arange(8, dtype=np.float64)[:, None]
        cols = np.arange(8, dtype=np.float64)[None, :]
        data = 3.0 * cols + 5.0 * rows
        source = metrics.Raster(data, 0.0, 8.0, 1.0, 1.0)
        values, _ = metrics.resample_to_grid(
            source, origin_x=0.5, origin_y=7.5, pixel_size=1.0, width=4, height=4
        )
        # Output cell (0,0) is centred at world (1.0, 7.0), whose continuous
        # source coordinate is (col 1.0, row 1.0) = (0.5, 0.5) in index terms.
        expected = 3.0 * (np.arange(4) + 0.5)[None, :] + 5.0 * (np.arange(4) + 0.5)[:, None]
        np.testing.assert_allclose(values[:, :, 0], expected, atol=1e-9)

    def test_nodata_does_not_bleed_into_a_bilinear_tap(self):
        data = np.full((4, 4), 10.0)
        data[1, 1] = np.nan
        source = metrics.Raster(data, 0.0, 4.0, 1.0, 1.0, nodata=np.nan)
        values, valid = metrics.resample_to_grid(
            source, origin_x=1.5, origin_y=2.5, pixel_size=1.0, width=1, height=1
        )
        # The single output cell straddles the NaN; the three valid taps are all
        # 10.0, so the renormalised value is 10.0 and the cell stays valid.
        np.testing.assert_allclose(values[0, 0, 0], 10.0)
        self.assertTrue(valid[0, 0])

    def test_a_cell_with_no_valid_tap_is_invalid(self):
        data = np.full((4, 4), np.nan)
        source = metrics.Raster(data, 0.0, 4.0, 1.0, 1.0, nodata=np.nan)
        _, valid = metrics.resample_to_grid(
            source, origin_x=1.5, origin_y=2.5, pixel_size=1.0, width=1, height=1
        )
        self.assertFalse(valid[0, 0])

    def test_nearest_neighbour_takes_the_containing_cell(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        source = metrics.Raster(data, 0.0, 2.0, 1.0, 1.0)
        values, _ = metrics.resample_to_grid(
            source, origin_x=0.0, origin_y=2.0, pixel_size=1.0, width=2, height=2, order=0
        )
        np.testing.assert_allclose(values[:, :, 0], data)

    def test_upsampling_by_two_still_marks_every_cell_valid(self):
        data = np.arange(4, dtype=np.float64).reshape(2, 2)
        source = metrics.Raster(data, 0.0, 2.0, 1.0, 1.0)
        values, valid = metrics.resample_to_grid(
            source, origin_x=0.0, origin_y=2.0, pixel_size=0.5, width=4, height=4
        )
        self.assertTrue(valid.all())
        # Corners of the padded grid clamp to the edge values, not to zero.
        self.assertEqual(values[0, 0, 0], 0.0)
        self.assertEqual(values[-1, -1, 0], 3.0)


class CommonGridTest(unittest.TestCase):
    def test_picks_the_finer_pixel_size(self):
        a = metrics.Raster(np.zeros((10, 10)), 0.0, 10.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((20, 20)), 0.0, 10.0, 0.5, 0.5)
        _, _, pixel, _, _ = metrics.common_grid(a, b)
        self.assertAlmostEqual(pixel, 0.5)

    def test_covers_only_the_intersection(self):
        a = metrics.Raster(np.zeros((10, 10)), 0.0, 10.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((10, 10)), 5.0, 10.0, 1.0, 1.0)
        ox, oy, _, width, height = metrics.common_grid(a, b)
        self.assertEqual((ox, oy), (5.0, 10.0))
        self.assertEqual((width, height), (5, 10))

    def test_coarsens_by_an_integer_when_over_the_cell_budget(self):
        a = metrics.Raster(np.zeros((1000, 1000)), 0.0, 1000.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((1000, 1000)), 0.0, 1000.0, 1.0, 1.0)
        _, _, pixel, width, height = metrics.common_grid(a, b, max_cells=250_000)
        self.assertEqual(pixel, 2.0)  # 1000x1000 over budget, 500x500 fits
        self.assertEqual((width, height), (500, 500))

    def test_rejects_non_overlapping_rasters(self):
        a = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((4, 4)), 100.0, 100.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            metrics.common_grid(a, b)

    def test_a_crs_mismatch_is_an_error_not_a_comparison(self):
        # Two grids whose coordinate values happen to overlap but which describe
        # different worlds. Silently comparing them is the failure this guards.
        a = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0, epsg=32632)
        b = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0, epsg=3857)
        with self.assertRaises(ValueError) as raised:
            metrics.common_grid(a, b)
        self.assertIn("32632", str(raised.exception))
        self.assertIn("3857", str(raised.exception))

    def test_an_unknown_crs_is_allowed_through(self):
        # A hand-built Raster in a test carries no EPSG; that is "not declared",
        # which is different from "declared and different".
        a = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0, epsg=32632)
        self.assertEqual(metrics.common_grid(a, b)[3:], (4, 4))


class AlignedReferenceTest(unittest.TestCase):
    """The CRS gate in front of every raster comparison, on real GeoTIFFs."""

    @staticmethod
    def _write(path, *, epsg, value, transform, size=(32, 32), bands=1):
        import rasterio  # noqa: PLC0415

        data = np.full((bands, size[1], size[0]), value, dtype="float32")
        with rasterio.open(
            str(path), "w", driver="GTiff", height=size[1], width=size[0],
            count=bands, dtype="float32", crs=f"EPSG:{epsg}",
            transform=transform, nodata=float("nan"),
        ) as dataset:
            dataset.write(data)

    def _skip_without_rasterio(self):
        try:
            import rasterio  # noqa: F401, PLC0415
        except ImportError:  # pragma: no cover - rasterio is in the workflow
            self.skipTest("rasterio is not installed")

    def test_a_matching_crs_is_read_without_a_warp(self):
        self._skip_without_rasterio()
        from rasterio.transform import from_origin  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.tif"
            self._write(path, epsg=32632, value=7.0, transform=from_origin(0, 32, 1, 1))
            ours = metrics.Raster(np.zeros((32, 32)), 0.0, 32.0, 1.0, 1.0, epsg=32632)
            reference, provenance = metrics.aligned_reference(str(path), ours, bounds=None)
        self.assertFalse(provenance["reprojected"])
        self.assertEqual(provenance["reference_epsg"], 32632)
        self.assertAlmostEqual(float(np.nanmax(reference.band(0))), 7.0)

    def test_a_reference_without_a_crs_against_ours_with_one_is_an_error(self):
        self._skip_without_rasterio()
        from rasterio.transform import from_origin  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.tif"
            import rasterio  # noqa: PLC0415

            with rasterio.open(
                str(path), "w", driver="GTiff", height=32, width=32, count=1,
                dtype="float32", transform=from_origin(0, 32, 1, 1),
            ) as dataset:
                dataset.write(np.full((1, 32, 32), 1.0, dtype="float32"))
            ours = metrics.Raster(np.zeros((32, 32)), 0.0, 32.0, 1.0, 1.0, epsg=32632)
            with self.assertRaises(ValueError):
                metrics.aligned_reference(str(path), ours, bounds=None)

    def test_a_different_crs_is_reprojected_onto_ours(self):
        self._skip_without_rasterio()
        import rasterio  # noqa: PLC0415
        from rasterio.transform import from_origin  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            utm = Path(tmp) / "ours.tif"
            mercator = Path(tmp) / "ref.tif"
            transform = from_origin(500000.0, 5500000.0, 1.0, 1.0)
            self._write(utm, epsg=32632, value=3.0, transform=transform)
            # Build the 3857 copy by warping the UTM one, so both describe the same
            # ground and the only difference is the datum of the grid.
            from rasterio.warp import Resampling, calculate_default_transform, reproject

            with rasterio.open(str(utm)) as src:
                dst_transform, width, height = calculate_default_transform(
                    src.crs, "EPSG:3857", src.width, src.height, *src.bounds
                )
                payload = np.full((height, width), np.nan, dtype="float32")
                reproject(
                    rasterio.band(src, 1), payload,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs="EPSG:3857",
                    resampling=Resampling.bilinear, dst_nodata=np.nan,
                )
            with rasterio.open(
                str(mercator), "w", driver="GTiff", height=height, width=width,
                count=1, dtype="float32", crs="EPSG:3857",
                transform=dst_transform, nodata=float("nan"),
            ) as dataset:
                dataset.write(payload, 1)

            ours = metrics.read_geotiff(str(utm))
            reference, provenance = metrics.aligned_reference(
                str(mercator), ours, bounds=ours.extent()
            )
            comparison = metrics.compare_dem_to_reference(ours, reference)

        self.assertTrue(provenance["reprojected"])
        self.assertEqual(reference.epsg, ours.epsg)
        # Same constant both sides: round-tripping the warp must not invent a bias.
        self.assertAlmostEqual(comparison["cells"], 32 * 32, delta=64)
        self.assertAlmostEqual(comparison["median_m"], 0.0, delta=0.02)


class ErosionTest(unittest.TestCase):
    def test_radius_zero_is_the_identity(self):
        mask = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(metrics.erosion(mask, 0), mask)

    def test_radius_one_removes_a_one_cell_border(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        eroded = metrics.erosion(mask, 1)
        expected = np.zeros((5, 5), dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(eroded, expected)

    def test_a_solid_block_survives_an_interior_erosion(self):
        mask = np.ones((7, 7), dtype=bool)
        self.assertTrue(metrics.erosion(mask, 1).any())


class ConvexHullTest(unittest.TestCase):
    def test_a_rectangle_hull_fills_a_missing_centre(self):
        mask = np.ones((10, 10), dtype=bool)
        mask[4:6, 4:6] = False
        hull = metrics.convex_hull_mask(mask)
        # Every cell of a rectangle's hull is inside; the hole must be covered.
        self.assertTrue(hull.all())

    def test_an_empty_mask_has_no_hull(self):
        self.assertFalse(metrics.convex_hull_mask(np.zeros((4, 4), dtype=bool)).any())

    def test_a_triangle_hull_excludes_the_corner_outside_it(self):
        # A right triangle with vertices at the top-left, bottom-left and
        # bottom-right cell centres. The bottom-LEFT corner lies outside it.
        mask = np.zeros((20, 20), dtype=bool)
        for y in range(20):
            mask[y, y:] = True
        hull = metrics.convex_hull_mask(mask)
        self.assertFalse(hull[19, 0])
        self.assertTrue(hull[19, 19])
        self.assertTrue(hull[0, 0])
        # The hull of a triangle holds every cell of the triangle itself.
        self.assertTrue(np.all(hull[mask]))

    def test_two_cells_are_kept_as_is(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = mask[2, 2] = True
        hull = metrics.convex_hull_mask(mask)
        self.assertEqual(int(hull.sum()), 2)


class HoleStatsTest(unittest.TestCase):
    def test_no_holes_in_a_full_mask(self):
        stats = metrics.hole_stats(np.ones((10, 10), dtype=bool))
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["area_px"], 0)

    def test_a_border_gap_is_not_a_hole(self):
        valid = np.ones((10, 10), dtype=bool)
        valid[:, 0] = False
        stats = metrics.hole_stats(valid)
        self.assertEqual(stats["count"], 0)

    def test_one_interior_block_counts_once(self):
        valid = np.ones((10, 10), dtype=bool)
        valid[4:6, 4:6] = False
        stats = metrics.hole_stats(valid, min_px=1)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["area_px"], 4)
        self.assertEqual(stats["largest_px"], 4)

    def test_two_separated_blocks_count_twice(self):
        valid = np.ones((10, 10), dtype=bool)
        valid[1:3, 1:3] = False
        valid[7:9, 7:9] = False
        stats = metrics.hole_stats(valid, min_px=1)
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["area_px"], 8)

    def test_a_diagonal_run_is_two_components_under_4_connectivity(self):
        valid = np.ones((5, 5), dtype=bool)
        valid[1, 1] = False
        valid[2, 2] = False
        stats = metrics.hole_stats(valid, min_px=1)
        self.assertEqual(stats["count"], 2)

    def test_min_px_filters_specks(self):
        valid = np.ones((10, 10), dtype=bool)
        valid[5, 5] = False
        self.assertEqual(metrics.hole_stats(valid, min_px=4)["count"], 0)
        self.assertEqual(metrics.hole_stats(valid, min_px=1)["count"], 1)

    def test_interior_restricts_to_the_hull(self):
        valid = np.zeros((10, 10), dtype=bool)
        valid[2:8, 2:8] = True
        valid[4, 4] = False
        interior = metrics.erosion(metrics.convex_hull_mask(valid), 1)
        stats = metrics.hole_stats(valid, interior=interior, min_px=1)
        self.assertEqual(stats["count"], 1)
        # Outside the hull is not a hole; only cells inside count as considered.
        self.assertEqual(stats["area_px"], 1)


class GradientEnergyTest(unittest.TestCase):
    def test_a_flat_image_has_no_energy(self):
        self.assertEqual(metrics.gradient_energy(np.full((16, 16), 0.5)), 0.0)

    def test_energy_falls_under_blur(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        sharp = metrics.gradient_energy(image)
        blurred = metrics.gradient_energy(metrics._box_blur(image, 2))
        self.assertGreater(sharp, blurred)

    def test_energy_ignores_holes(self):
        image = np.zeros((16, 16))
        image[:, 8:] = 1.0
        valid = np.ones((16, 16), dtype=bool)
        valid[0, 0] = False
        # The two means average over one cell fewer, so they agree to within the
        # single cell's own contribution rather than bit-for-bit.
        self.assertAlmostEqual(
            metrics.gradient_energy(image, valid=valid),
            metrics.gradient_energy(image),
            delta=1e-3,
        )

    def test_a_tiny_image_has_no_energy(self):
        self.assertEqual(metrics.gradient_energy(np.zeros((2, 2))), 0.0)


class MtfTest(unittest.TestCase):
    def test_a_sharp_random_image_reports_a_ratio_near_one(self):
        rng = np.random.default_rng(7)
        image = rng.random((128, 128))
        result = metrics.mtf50_effective_resolution(image, pixel_size_m=0.02)
        self.assertTrue(result["textured"])
        # White noise has almost no spatial correlation, so any blur halves its
        # gradient energy immediately: the ratio stays under two.
        self.assertLess(result["ratio"], 2.0)
        self.assertGreater(result["ratio"], 0.0)

    def test_a_pre_blurred_image_reports_a_larger_ratio(self):
        rng = np.random.default_rng(11)
        image = rng.random((128, 128))
        sharp = metrics.mtf50_effective_resolution(image, pixel_size_m=0.02)
        soft = metrics.mtf50_effective_resolution(metrics._box_blur(image, 2), pixel_size_m=0.02)
        self.assertGreater(soft["ratio"], sharp["ratio"])

    def test_effective_resolution_is_the_cell_size_times_the_ratio(self):
        rng = np.random.default_rng(3)
        image = rng.random((64, 64))
        one = metrics.mtf50_effective_resolution(image, pixel_size_m=0.01)
        two = metrics.mtf50_effective_resolution(image, pixel_size_m=0.02)
        # The ratio is a property of the image alone; only the metre conversion
        # follows the cell size.
        self.assertAlmostEqual(two["ratio"], one["ratio"], places=12)
        self.assertAlmostEqual(
            two["effective_resolution_m"], 2.0 * one["effective_resolution_m"], places=12
        )

    def test_a_flat_image_is_not_textured(self):
        result = metrics.mtf50_effective_resolution(np.full((32, 32), 0.25), pixel_size_m=0.02)
        self.assertFalse(result["textured"])
        self.assertFalse(result["measured"])
        self.assertIsNone(result["ratio"])

    def test_a_search_that_never_halves_is_unmeasured_not_capped(self):
        # A very smooth image whose energy only falls a little under the whole
        # search: the answer is "no half-energy point inside the cap", never the
        # cap itself dressed up as a measurement.
        image = np.tile(np.arange(64, dtype=np.float64), (64, 1))
        result = metrics.mtf50_effective_resolution(image, pixel_size_m=0.05, max_blur_radius=2)
        self.assertFalse(result["measured"])
        self.assertIsNone(result["ratio"])
        self.assertIsNone(result["effective_resolution_m"])
        self.assertIn("cap", result["reason"])

    def test_a_non_positive_pixel_size_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.mtf50_effective_resolution(np.ones((8, 8)), pixel_size_m=0.0)

    @staticmethod
    def _blurred_step(sigma_px: float, size: int = 1024) -> np.ndarray:
        """A straight step edge blurred by a KNOWN Gaussian, pixel-area sampled.

        Each pixel holds the integral of the blurred edge over its own width,
        which is what a sensor records, so a sharp edge (sigma_px = 0) already
        carries the 1/sqrt(12) of a cell that area sampling implies.

        The answer is known by construction: Sobel gradient energy of an edge
        falls as 1/sigma_total, so it halves when the total sigma doubles, and
        the effective resolution in cells is sqrt(1 + 12*sigma_px^2).
        """
        mid = np.arange(size, dtype=np.float64) - size / 2.0 + 0.5
        if sigma_px <= 0.0:
            row = (mid < 0).astype(np.float64)
        else:
            row = 0.5 * np.vectorize(math.erfc)(mid / (math.sqrt(2.0) * sigma_px))
        return np.tile(row, (16, 1))

    def test_a_known_gaussian_blur_is_recovered_analytically(self):
        # The claim the metric rests on: for a step blurred by a known Gaussian,
        # the ratio equals sqrt(1 + 12*sigma^2) to within 10%. Without this the
        # blur-to-metres conversion is a constant nobody checked.
        for sigma_px in (1.0, 2.0, 3.0, 4.0):
            with self.subTest(sigma_px=sigma_px):
                analytic = math.sqrt(1.0 + 12.0 * sigma_px * sigma_px)
                result = metrics.mtf50_effective_resolution(
                    self._blurred_step(sigma_px),
                    pixel_size_m=0.1,
                    max_blur_radius=60,
                )
                self.assertTrue(result["measured"])
                self.assertLess(abs(result["ratio"] / analytic - 1.0), 0.10)

    def test_the_analytic_ratio_converts_to_metres_through_the_cell(self):
        # 3 cells of Gaussian on a 0.05 m grid is 3.6 cells of effective
        # resolution, i.e. 0.18 m — derived from the mosaic's own cell, with no
        # dataset GSD anywhere in the arithmetic.
        result = metrics.mtf50_effective_resolution(
            self._blurred_step(1.0), pixel_size_m=0.05, max_blur_radius=60
        )
        analytic_ratio = math.sqrt(1.0 + 12.0 * 1.0)
        self.assertAlmostEqual(
            result["effective_resolution_m"],
            0.05 * result["ratio"],
            places=12,
        )
        self.assertLess(abs(result["effective_resolution_m"] - 0.05 * analytic_ratio), 0.05 * analytic_ratio * 0.10)

    def test_a_perfect_edge_reads_at_the_methods_floor(self):
        # The method's floor: the search starts at radius 1, so a perfect edge is
        # already past half energy there. It must report a measured ratio near 2,
        # never zero or a negative - and reporting ~2 for a perfect edge is the
        # documented, conservative floor.
        result = metrics.mtf50_effective_resolution(
            self._blurred_step(0.0), pixel_size_m=0.1, max_blur_radius=60
        )
        self.assertTrue(result["measured"])
        self.assertGreater(result["ratio"], 1.0)
        self.assertLess(result["ratio"], 2.5)


class PhaseCorrelationTest(unittest.TestCase):
    @staticmethod
    def _textured(size=128, seed=5):
        rng = np.random.default_rng(seed)
        image = rng.random((size, size))
        # Smooth a little so the correlation peak is well formed rather than a
        # single noisy cell.
        return metrics._box_blur(image, 1)

    def test_a_zero_shift_is_detected_as_zero(self):
        image = self._textured()
        shift = metrics.phase_correlation_shift(image, image)
        self.assertTrue(shift["in_bounds"])
        self.assertAlmostEqual(shift["dx"], 0.0, places=6)
        self.assertAlmostEqual(shift["dy"], 0.0, places=6)

    def test_an_integer_shift_is_recovered(self):
        image = self._textured()
        moved = np.roll(np.roll(image, 5, axis=1), -3, axis=0)
        shift = metrics.phase_correlation_shift(image, moved)
        self.assertTrue(shift["in_bounds"])
        self.assertAlmostEqual(shift["dx"], 5.0, places=2)
        self.assertAlmostEqual(shift["dy"], -3.0, places=2)

    def test_a_half_pixel_shift_is_recovered_sub_pixel(self):
        image = self._textured()
        # A cyclic half-pixel shift via the FFT is exactly representable and is
        # the sharpest test of the sub-pixel refinement.
        spectrum = np.fft.fft2(image)
        ky = np.fft.fftfreq(image.shape[0])[:, None]
        kx = np.fft.fftfreq(image.shape[1])[None, :]
        moved = np.real(np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (0.5 * kx + 0.0 * ky))))
        shift = metrics.phase_correlation_shift(image, moved)
        self.assertTrue(shift["in_bounds"])
        self.assertAlmostEqual(shift["dx"], 0.5, places=2)

    def test_an_offset_beyond_the_window_is_reported_out_of_bounds(self):
        image = self._textured(64)
        moved = np.roll(image, 25, axis=1)  # 25 > 0.25 * 64
        shift = metrics.phase_correlation_shift(image, moved, max_shift_fraction=0.1)
        self.assertFalse(shift["in_bounds"])

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.phase_correlation_shift(np.zeros((16, 16)), np.zeros((16, 17)))

    def test_a_tiny_array_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.phase_correlation_shift(np.zeros((4, 4)), np.zeros((4, 4)))

    def test_an_illumination_gradient_does_not_create_a_shift(self):
        image = self._textured()
        gradient = np.linspace(0.0, 0.5, image.shape[1])[None, :]
        shift = metrics.phase_correlation_shift(image, image + gradient)
        self.assertAlmostEqual(shift["dx"], 0.0, places=2)
        self.assertAlmostEqual(shift["dy"], 0.0, places=2)


class VoxelTest(unittest.TestCase):
    def test_one_point_per_voxel(self):
        points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]])
        down = metrics.voxel_downsample(points, 0.1)
        self.assertEqual(down.shape, (2, 3))

    def test_an_empty_cloud_stays_empty(self):
        self.assertEqual(metrics.voxel_downsample(np.zeros((0, 3)), 0.1).shape, (0, 3))

    def test_downsampling_is_deterministic(self):
        rng = np.random.default_rng(2)
        points = rng.random((500, 3))
        first = metrics.voxel_downsample(points, 0.05)
        second = metrics.voxel_downsample(points, 0.05)
        np.testing.assert_array_equal(first, second)

    def test_a_non_positive_voxel_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.voxel_downsample(np.zeros((1, 3)), 0.0)


class NearestNeighbourTest(unittest.TestCase):
    def test_distance_to_the_single_target(self):
        query = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        target = np.zeros((1, 3))
        np.testing.assert_allclose(metrics.nn_distances(query, target), [0.0, 5.0])

    def test_picks_the_nearest_of_several(self):
        query = np.array([[0.5, 0.0, 0.0]])
        target = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        np.testing.assert_allclose(metrics.nn_distances(query, target), [0.5])

    def test_a_small_block_gives_the_same_answer_as_a_large_one(self):
        # The block size is a memory bound, not part of the maths: a run split into
        # many small blocks must produce the same distances as one big block. The
        # limit is patched rather than passed, because it has to apply to the
        # internal calls in register_clouds too.
        rng = np.random.default_rng(9)
        query = rng.random((50, 3))
        target = rng.random((80, 3))
        original = metrics._NN_BLOCK_PAIRS
        try:
            metrics._NN_BLOCK_PAIRS = 80  # one row per block
            small = metrics.nn_distances(query, target)
            small_index = metrics._nn_indices(query, target)
            metrics._NN_BLOCK_PAIRS = 1 << 20
            large = metrics.nn_distances(query, target)
            large_index = metrics._nn_indices(query, target)
        finally:
            metrics._NN_BLOCK_PAIRS = original
        np.testing.assert_allclose(small, large, atol=1e-12)
        np.testing.assert_array_equal(small_index, large_index)

    def test_an_empty_query_gives_an_empty_result(self):
        self.assertEqual(metrics.nn_distances(np.zeros((0, 3)), np.zeros((3, 3))).size, 0)

    def test_an_empty_target_gives_infinity(self):
        result = metrics.nn_distances(np.zeros((2, 3)), np.zeros((0, 3)))
        self.assertTrue(np.isinf(result).all())


class AccuracyCompletenessTest(unittest.TestCase):
    def test_identical_clouds_score_perfectly(self):
        rng = np.random.default_rng(4)
        cloud = rng.random((200, 3)) * 0.2
        result = metrics.accuracy_completeness(cloud, cloud, [0.02, 0.05, 0.10])
        self.assertAlmostEqual(result["accuracy"]["0.02"], 1.0)
        self.assertAlmostEqual(result["completeness"]["0.02"], 1.0)
        self.assertAlmostEqual(result["f_score_5cm"], 1.0)
        self.assertLess(result["accuracy_median_m"], 0.02)

    def test_a_rigid_offset_fails_the_tight_tolerance(self):
        # A 3 cm shift is outside a 2 cm tolerance. Voxel-normalisation at the
        # same 1 cm voxel lets a few per-cent of points still find a reference
        # neighbour inside 2 cm (they sit near a voxel boundary), which is the
        # expected behaviour of the ETH3D definition rather than a bug, so the
        # assertion is "mostly outside" rather than "exactly zero".
        rng = np.random.default_rng(6)
        reference = rng.random((300, 3)) * 0.5
        shifted = reference + np.array([0.03, 0.0, 0.0])
        # register=False: this test is about the score itself, not the alignment.
        result = metrics.accuracy_completeness(
            shifted, reference, [0.02, 0.05, 0.10], register=False
        )
        self.assertLess(result["accuracy"]["0.02"], 0.2)
        self.assertAlmostEqual(result["accuracy"]["0.05"], 1.0)
        self.assertAlmostEqual(result["accuracy_median_m"], 0.03, delta=0.01)

    def test_a_shifted_cluster_keeps_its_completeness(self):
        # Translating the whole cloud out of the volume loses accuracy but the
        # reference is still inside the shifted cloud's union of voxels, so
        # completeness at a wide tolerance stays high. This pins the direction
        # of each metric.
        reference = np.stack(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5),
                                         np.linspace(0, 1, 5)), axis=-1).reshape(-1, 3)
        shifted = reference + 0.05
        # register=False: with registration on, the whole point is that such a
        # shift gets removed, which the registration tests cover.
        result = metrics.accuracy_completeness(shifted, reference, [0.10], register=False)
        self.assertAlmostEqual(result["accuracy"]["0.1"], 1.0)
        self.assertAlmostEqual(result["completeness"]["0.1"], 1.0)

    def test_an_empty_cloud_scores_zero_not_an_error(self):
        result = metrics.accuracy_completeness(np.zeros((0, 3)), np.ones((5, 3)), [0.05])
        self.assertEqual(result["accuracy"]["0.05"], 0.0)
        self.assertEqual(result["completeness"]["0.05"], 0.0)
        self.assertEqual(result["reconstructed"], 0)

    def test_voxel_normalisation_does_not_reward_density(self):
        sparse = np.zeros((1, 3))
        dense = np.repeat(np.zeros((1, 3)), 100, axis=0)
        reference = np.zeros((1, 3))
        a = metrics.accuracy_completeness(sparse, reference, [0.01])
        b = metrics.accuracy_completeness(dense, reference, [0.01])
        self.assertEqual(a["accuracy"]["0.01"], b["accuracy"]["0.01"])


class UmeyamaTest(unittest.TestCase):
    @staticmethod
    def _cloud(n=200, seed=31):
        rng = np.random.default_rng(seed)
        return rng.random((n, 3)) * 2.0

    def test_a_known_similarity_is_recovered(self):
        source = self._cloud()
        angle = 0.4
        rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = 2.5
        translation = np.array([1.0, -2.0, 0.5])
        target = scale * (source @ rotation.T) + translation

        got_scale, got_rotation, got_translation = metrics.umeyama_similarity(source, target)
        self.assertAlmostEqual(got_scale, scale, places=9)
        np.testing.assert_allclose(got_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(got_translation, translation, atol=1e-9)

    def test_a_reflection_is_not_returned(self):
        # A mirrored point set has no proper rotation that maps it onto itself, so
        # the fit must still come out with det(R) = +1 rather than a reflection.
        source = self._cloud()
        target = source * np.array([1.0, 1.0, -1.0])
        _, rotation, _ = metrics.umeyama_similarity(source, target)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_identity_input_gives_the_identity(self):
        source = self._cloud()
        scale, rotation, translation = metrics.umeyama_similarity(source, source)
        self.assertAlmostEqual(scale, 1.0, places=9)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-9)

    def test_two_points_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.umeyama_similarity(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_a_degenerate_source_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.umeyama_similarity(np.ones((5, 3)), np.zeros((5, 3)))

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.umeyama_similarity(np.zeros((4, 3)), np.zeros((5, 3)))


class ApplySimilarityTest(unittest.TestCase):
    def test_the_transform_is_scale_rotate_translate(self):
        points = np.array([[1.0, 0.0, 0.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        got = metrics.apply_similarity(2.0, rotation, np.array([0.0, 0.0, 5.0]), points)
        np.testing.assert_allclose(got, [[0.0, 2.0, 5.0]])

    def test_an_empty_cloud_stays_empty(self):
        got = metrics.apply_similarity(1.0, np.eye(3), np.zeros(3), np.zeros((0, 3)))
        self.assertEqual(got.shape, (0, 3))


class RegisterCloudsTest(unittest.TestCase):
    def test_a_translated_and_scaled_copy_is_registered_back(self):
        rng = np.random.default_rng(41)
        source = rng.random((400, 3))
        target = 3.0 * source + np.array([5.0, -1.0, 2.0])
        aligned, info = metrics.register_clouds(source, target, voxel=0.02)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["scale"], 3.0, places=2)
        # After the fit the two clouds sit on top of each other.
        residual = metrics.nn_distances(aligned, metrics.voxel_downsample(target, 0.02))
        self.assertLess(float(np.median(residual)), 0.05)

    def test_two_clouds_already_aligned_stay_put(self):
        rng = np.random.default_rng(42)
        source = rng.random((300, 3))
        aligned, info = metrics.register_clouds(source, source, voxel=0.02)
        self.assertAlmostEqual(info["scale"], 1.0, places=3)
        np.testing.assert_allclose(aligned, source, atol=1e-6)

    def test_an_empty_cloud_is_reported_not_raised(self):
        aligned, info = metrics.register_clouds(np.zeros((0, 3)), np.ones((10, 3)), voxel=0.1)
        self.assertEqual(aligned.shape, (0, 3))
        self.assertFalse(info["converged"])

    def test_a_tiny_cloud_is_reported_not_raised(self):
        aligned, info = metrics.register_clouds(
            np.array([[0.0, 0.0, 0.0]]), np.ones((10, 3)), voxel=0.1
        )
        self.assertFalse(info["converged"])
        self.assertIn("reason", info)


class AccuracyCompletenessRegistrationTest(unittest.TestCase):
    def test_registration_removes_an_arbitrary_frame_difference(self):
        rng = np.random.default_rng(51)
        reference = rng.random((400, 3)) * 2.0
        # The reconstruction lives in its own frame: scaled, rotated, translated.
        angle = 0.7
        rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        reconstruction = 0.4 * (reference @ rotation.T) + np.array([10.0, 4.0, -3.0])

        without = metrics.accuracy_completeness(
            reconstruction, reference, [0.05], register=False
        )
        with_registration = metrics.accuracy_completeness(
            reconstruction, reference, [0.05], register=True
        )
        self.assertLess(without["accuracy"]["0.05"], 0.1)
        self.assertGreater(with_registration["accuracy"]["0.05"], 0.9)
        self.assertIn("alignment", with_registration)
        self.assertAlmostEqual(with_registration["alignment"]["scale"], 2.5, places=2)

    def test_registration_does_not_manufacture_a_perfect_score(self):
        # A reference deliberately missing half its surface: the reconstruction
        # cannot score full completeness no matter how well it is registered.
        rng = np.random.default_rng(52)
        reference = rng.random((400, 3))
        half = reference[reference[:, 0] < 0.5]
        result = metrics.accuracy_completeness(half, reference, [0.05])
        self.assertLess(result["completeness"]["0.05"], 0.8)


class DemDifferenceTest(unittest.TestCase):
    def test_a_constant_bias_is_the_median(self):
        a = np.full((4, 4), 10.5)
        b = np.full((4, 4), 10.0)
        result = metrics.dem_difference(a, b)
        self.assertAlmostEqual(result["median_m"], 0.5)
        self.assertAlmostEqual(result["nmad_m"], 0.0)
        self.assertEqual(result["cells"], 16)

    def test_a_constant_datum_offset_disappears_from_the_offset_removed_spread(self):
        # The case this exists for: our DEM in ellipsoidal heights against a
        # reference in a national orthometric datum. The raw median is metres, the
        # offset-removed spread is what the reconstruction actually did.
        rng = np.random.default_rng(21)
        reference = rng.normal(100.0, 3.0, (40, 40))
        noise = rng.normal(0.0, 0.05, (40, 40))
        ours = reference + 47.31 + noise
        result = metrics.dem_difference(ours, reference)
        self.assertAlmostEqual(result["median_m"], 47.31, delta=0.01)
        removed = result["offset_removed"]
        self.assertLess(abs(removed["nmad_m"]), 0.10)
        self.assertLess(abs(removed["p68_m"]), 0.10)
        # The raw percentile includes the constant, which is why it is not gated.
        self.assertGreater(result["p68_m"], 47.0)

    def test_the_offset_removed_percentiles_drop_the_constant(self):
        # delta = [0, 2, 10, 10] -> median 6, removed = [-6, -4, 4, 4]. The raw p95
        # carries the offset, the removed one does not.
        ours = np.array([[0.0, 2.0], [10.0, 10.0]])
        reference = np.zeros((2, 2))
        result = metrics.dem_difference(ours, reference)
        self.assertAlmostEqual(result["median_m"], 6.0)
        self.assertAlmostEqual(result["p95_m"], 10.0, places=6)
        self.assertLess(result["offset_removed"]["p95_m"], result["p95_m"])
        # |removed| = [6, 4, 4, 4], so the interpolated p95 lands just under the max
        # and well under the raw p95 of 10.
        self.assertLess(result["offset_removed"]["p95_m"], 6.5)

    def test_a_single_outlier_does_not_move_the_nmad(self):
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        a[0, 0] = 100.0
        result = metrics.dem_difference(a, b)
        self.assertAlmostEqual(result["median_m"], 0.0)
        self.assertAlmostEqual(result["nmad_m"], 0.0)

    def test_nan_cells_are_excluded(self):
        a = np.array([[1.0, np.nan]])
        b = np.zeros((1, 2))
        self.assertEqual(metrics.dem_difference(a, b)["cells"], 1)

    def test_an_explicit_valid_mask_narrows_the_comparison(self):
        a = np.ones((2, 2))
        b = np.zeros((2, 2))
        valid = np.array([[True, False], [False, False]])
        result = metrics.dem_difference(a, b, valid=valid)
        self.assertEqual(result["cells"], 1)

    def test_no_overlap_reports_none(self):
        result = metrics.dem_difference(np.full((2, 2), np.nan), np.zeros((2, 2)))
        self.assertEqual(result["cells"], 0)
        self.assertIsNone(result["median_m"])

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.dem_difference(np.zeros((2, 2)), np.zeros((2, 3)))


class ProjectionTest(unittest.TestCase):
    @staticmethod
    def identity_pose():
        return np.eye(4)

    def test_a_point_on_the_axis_lands_at_the_principal_point(self):
        calibration = {"type": "frame", "width": 100, "height": 50, "f": 50.0}
        pixels, ok = metrics.project_pixels(
            np.array([[0.0, 0.0, 10.0]]), self.identity_pose(), calibration
        )
        self.assertTrue(ok[0])
        np.testing.assert_allclose(pixels[0], [50.0, 25.0])

    def test_a_known_offset_projects_to_the_hand_computed_pixel(self):
        calibration = {"type": "frame", "width": 100, "height": 50, "f": 50.0}
        # X/Z = 0.2 -> u = 50 + 0.2*50 = 60 ; Y/Z = -0.1 -> v = 25 - 5 = 20
        pixels, _ = metrics.project_pixels(
            np.array([[2.0, -1.0, 10.0]]), self.identity_pose(), calibration
        )
        np.testing.assert_allclose(pixels[0], [60.0, 20.0])

    def test_a_lower_case_frame_project_reprojects_under_a_pixel(self):
        # The regression this guards: the writer emits ``type="frame"`` but the
        # projection used to compare against "Frame", so every frame camera fell
        # through to the fisheye branch and a pinhole scene re-projected ~160 px
        # off. A lowercase frame project must come back exact.
        calibration = {"type": "frame", "width": 6000, "height": 4000, "f": 5000.0}
        rng = np.random.default_rng(7)
        points = np.column_stack([
            rng.uniform(-40.0, 40.0, 200),
            rng.uniform(-25.0, 25.0, 200),
            np.full(200, 100.0),
        ])
        pose = np.eye(4)
        # Yaw the camera a little; every point stays well in front of it, so the
        # comparison is about the projection model and not about culling.
        yaw = math.radians(12.0)
        pose[:3, :3] = np.array(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ]
        )
        pose[:3, 3] = [0.0, 0.0, 0.0]
        pixels, ok = metrics.project_pixels(points, pose, calibration)
        self.assertTrue(ok.all())
        # Compare against an explicit pinhole computation with the same pose.
        cam = points @ pose[:3, :3].T + pose[:3, 3]
        expected = np.column_stack([
            calibration["width"] * 0.5 + calibration["f"] * cam[:, 0] / cam[:, 2],
            calibration["height"] * 0.5 + calibration["f"] * cam[:, 1] / cam[:, 2],
        ])
        error = np.linalg.norm(pixels - expected, axis=1)
        self.assertLess(float(error.max()), 1.0)

    def test_the_type_is_read_case_insensitively(self):
        point = np.array([[2.0, -1.0, 10.0]])
        lower, _ = metrics.project_pixels(
            point, self.identity_pose(), {"type": "frame", "width": 100, "height": 50, "f": 50.0}
        )
        upper, _ = metrics.project_pixels(
            point, self.identity_pose(), {"type": " Frame ", "width": 100, "height": 50, "f": 50.0}
        )
        np.testing.assert_allclose(lower, upper)

    def test_an_unknown_camera_type_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.project_pixels(
                np.array([[0.0, 0.0, 1.0]]),
                self.identity_pose(),
                {"type": "spherical", "width": 10, "height": 10, "f": 5.0},
            )

    def test_the_principal_point_offset_moves_the_image_of_the_axis(self):
        calibration = {"type": "frame", "width": 100, "height": 50, "f": 50.0, "cx": 4.0, "cy": -3.0}
        pixels, _ = metrics.project_pixels(
            np.array([[0.0, 0.0, 10.0]]), self.identity_pose(), calibration
        )
        np.testing.assert_allclose(pixels[0], [54.0, 22.0])

    def test_radial_distortion_moves_a_point_outward_for_positive_k1(self):
        straight = {"type": "frame", "width": 100, "height": 100, "f": 50.0}
        bent = dict(straight, k1=0.5)
        point = np.array([[5.0, 0.0, 10.0]])
        plain, _ = metrics.project_pixels(point, self.identity_pose(), straight)
        distorted, _ = metrics.project_pixels(point, self.identity_pose(), bent)
        self.assertGreater(distorted[0, 0], plain[0, 0])

    def test_higher_order_tangential_terms_fold_into_the_brown_model(self):
        # p3/p4 weight the p1/p2 template by r^2: q1 = p1 + p3 r^2,
        # q2 = p2 + p4 r^2. Explicit zeros must leave the projection untouched;
        # a non-zero pair must move the pixel by the hand-computed amount.
        straight = {
            "type": "frame",
            "width": 100,
            "height": 100,
            "f": 50.0,
            "p1": 1e-3,
            "p2": -2e-3,
        }
        zeroed = dict(straight, p3=0.0, p4=0.0)
        folded = dict(straight, p3=2e-4, p4=-1e-4)
        point = np.array([[5.0, 3.0, 10.0]])
        plain, _ = metrics.project_pixels(point, self.identity_pose(), straight)
        explicit_zero, _ = metrics.project_pixels(point, self.identity_pose(), zeroed)
        np.testing.assert_array_equal(explicit_zero, plain)
        bent, _ = metrics.project_pixels(point, self.identity_pose(), folded)
        x, y = 0.5, 0.3
        r2 = x * x + y * y
        q1 = straight["p1"] + folded["p3"] * r2
        q2 = straight["p2"] + folded["p4"] * r2
        xd = x + q1 * (r2 + 2.0 * x * x) + 2.0 * q2 * x * y
        yd = y + q2 * (r2 + 2.0 * y * y) + 2.0 * q1 * x * y
        self.assertAlmostEqual(bent[0, 0], 50.0 + xd * 50.0)
        self.assertAlmostEqual(bent[0, 1], 50.0 + yd * 50.0)

    def test_a_point_behind_the_camera_is_not_projected(self):
        calibration = {"type": "frame", "width": 100, "height": 100, "f": 50.0}
        pixels, ok = metrics.project_pixels(
            np.array([[0.0, 0.0, -1.0]]), self.identity_pose(), calibration
        )
        self.assertFalse(ok[0])
        self.assertTrue(np.isnan(pixels[0, 0]))

    def test_the_pose_is_applied_world_to_camera(self):
        calibration = {"type": "frame", "width": 100, "height": 100, "f": 50.0}
        pose = np.eye(4)
        pose[:3, 3] = [-10.0, 0.0, 0.0]  # the camera sits at x = +10
        pixels, _ = metrics.project_pixels(np.array([[10.0, 0.0, 5.0]]), pose, calibration)
        self.assertAlmostEqual(pixels[0, 0], 50.0)

    def test_fisheye_uses_an_equidistant_model(self):
        calibration = {"type": "fisheye", "width": 100, "height": 100, "f": 25.0}
        # 45 degrees off axis: theta = pi/4, so x_s = theta and u = 50 + 25*pi/4.
        pixels, ok = metrics.project_pixels(
            np.array([[10.0, 0.0, 10.0]]), self.identity_pose(), calibration
        )
        self.assertTrue(ok[0])
        self.assertAlmostEqual(pixels[0, 0], 50.0 + 25.0 * math.pi / 4.0)

    def test_an_empty_point_set_projects_to_nothing(self):
        calibration = {"type": "frame", "width": 10, "height": 10, "f": 5.0}
        pixels, ok = metrics.project_pixels(np.zeros((0, 3)), self.identity_pose(), calibration)
        self.assertEqual(pixels.shape, (0, 2))
        self.assertEqual(ok.size, 0)


class ReprojectionStatsTest(unittest.TestCase):
    def test_identical_pixels_have_zero_error(self):
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = metrics.reprojection_stats(observed, observed)
        self.assertEqual(stats["pairs"], 2)
        self.assertAlmostEqual(stats["rms_px"], 0.0)

    def test_a_known_offset_is_the_rms(self):
        observed = np.array([[0.0, 0.0], [0.0, 0.0]])
        projected = np.array([[3.0, 4.0], [3.0, 4.0]])
        stats = metrics.reprojection_stats(observed, projected)
        self.assertAlmostEqual(stats["rms_px"], 5.0)

    def test_non_finite_projections_are_dropped(self):
        observed = np.array([[0.0, 0.0], [0.0, 0.0]])
        projected = np.array([[np.nan, np.nan], [1.0, 0.0]])
        stats = metrics.reprojection_stats(observed, projected)
        self.assertEqual(stats["pairs"], 1)
        self.assertAlmostEqual(stats["rms_px"], 1.0)

    def test_nothing_left_reports_none(self):
        observed = np.array([[0.0, 0.0]])
        projected = np.array([[np.nan, np.nan]])
        stats = metrics.reprojection_stats(observed, projected)
        self.assertEqual(stats["pairs"], 0)
        self.assertIsNone(stats["rms_px"])

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.reprojection_stats(np.zeros((2, 2)), np.zeros((3, 2)))


class PlyReaderTest(unittest.TestCase):
    def test_binary_ply_with_colours_is_read_by_stride(self):
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        body = b""
        for index in range(3):
            body += struct.pack("<ddd", float(index), float(index) * 2, float(index) * 3)
            body += struct.pack("<BBB", 10, 20, 30)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as handle:
            handle.write(header.encode("ascii") + body)
            path = handle.name
        try:
            points = metrics.read_ply_points(path)
            np.testing.assert_allclose(points, [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
            subset = metrics.read_ply_points(path, max_points=2)
            self.assertEqual(subset.shape, (2, 3))
            np.testing.assert_allclose(subset[1], [2, 4, 6])
        finally:
            os.unlink(path)

    def test_ascii_ply_is_read(self):
        text = (
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            "1 2 3\n4 5 6\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as handle:
            handle.write(text)
            path = handle.name
        try:
            np.testing.assert_allclose(metrics.read_ply_points(path), [[1, 2, 3], [4, 5, 6]])
        finally:
            os.unlink(path)

    def test_a_header_without_xyz_is_rejected(self):
        header = (
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float a\nproperty float b\nproperty float c\nend_header\n0 0 0\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as handle:
            handle.write(header)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                metrics.read_ply_points(path)
        finally:
            os.unlink(path)

    def test_a_non_ply_file_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as handle:
            handle.write("not a ply\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                metrics.read_ply_points(path)
        finally:
            os.unlink(path)


class ObjReaderTest(unittest.TestCase):
    def test_triangles_are_read_and_quads_are_fanned(self):
        text = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3\nf 1 2 3 4\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as handle:
            handle.write(text)
            path = handle.name
        try:
            triangles = metrics.read_obj_triangles(path)
            # One explicit triangle plus two from the quad fan.
            self.assertEqual(triangles.shape, (3, 3, 3))
            np.testing.assert_allclose(triangles[0, 0], [0, 0, 0])
        finally:
            os.unlink(path)

    def test_slash_separated_face_indices_are_understood(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as handle:
            handle.write(text)
            path = handle.name
        try:
            self.assertEqual(metrics.read_obj_triangles(path).shape, (1, 3, 3))
        finally:
            os.unlink(path)

    def test_an_obj_without_faces_is_empty(self):
        text = "v 0 0 0\nv 1 0 0\n"
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as handle:
            handle.write(text)
            path = handle.name
        try:
            self.assertEqual(metrics.read_obj_triangles(path).shape, (0, 3, 3))
        finally:
            os.unlink(path)


class DecimateMaskTest(unittest.TestCase):
    def test_a_small_mask_is_returned_unchanged(self):
        mask = np.ones((10, 10), dtype=bool)
        small, stride = metrics.decimate_mask(mask, 1200)
        self.assertEqual(stride, 1)
        np.testing.assert_array_equal(small, mask)

    def test_stride_is_derived_from_the_long_side(self):
        mask = np.ones((240, 60), dtype=bool)
        small, stride = metrics.decimate_mask(mask, 120)
        self.assertEqual(stride, 2)
        self.assertEqual(small.shape, (120, 30))

    def test_any_valid_cell_keeps_the_block_valid(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        small, stride = metrics.decimate_mask(mask, 2)
        self.assertEqual(stride, 2)
        self.assertTrue(small[0, 0])

    def test_a_block_of_all_invalid_stays_invalid(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        small, _ = metrics.decimate_mask(mask, 2)
        self.assertFalse(small[1, 1])


class OrthoMetricsTest(unittest.TestCase):
    @staticmethod
    def _ortho(size=64, *, alpha_hole=None, seed=13):
        rng = np.random.default_rng(seed)
        data = np.zeros((size, size, 4), dtype=np.uint8)
        data[:, :, :3] = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        data[:, :, 3] = 255
        if alpha_hole is not None:
            for (r0, r1, c0, c1) in alpha_hole:
                data[r0:r1, c0:c1, 3] = 0
        return metrics.Raster(data, 0.0, float(size), 0.05, 0.05)

    def test_full_coverage_reports_one(self):
        result = metrics.ortho_metrics(self._ortho())
        self.assertAlmostEqual(result["covered_fraction"], 1.0)
        self.assertEqual(result["holes_inside_hull"], 0)

    def test_an_interior_hole_is_counted_with_its_area(self):
        result = metrics.ortho_metrics(self._ortho(64, alpha_hole=[(20, 30, 20, 30)]))
        self.assertAlmostEqual(result["covered_fraction"], (64 * 64 - 100) / (64 * 64))
        self.assertGreaterEqual(result["holes_inside_hull"], 1)
        self.assertGreater(result["hole_area_m2"], 0.0)

    def test_a_border_gap_is_not_a_hole(self):
        result = metrics.ortho_metrics(self._ortho(64, alpha_hole=[(0, 64, 0, 4)]))
        self.assertEqual(result["holes_inside_hull"], 0)

    def test_the_alpha_channel_not_the_colour_decides_coverage(self):
        # An all-black RGB block with a set alpha is a measurement of a black
        # surface, not a gap.
        ortho = self._ortho(32)
        ortho.data[8:16, 8:16, :3] = 0
        result = metrics.ortho_metrics(ortho)
        self.assertAlmostEqual(result["covered_fraction"], 1.0)

    def test_effective_resolution_is_reported_against_the_mosaic_cell(self):
        # The metric must not depend on a dataset's nominal GSD: 0.05 m cells mean
        # effective_resolution_m = 0.05 * ratio, whatever the imagery's own GSD was.
        result = metrics.ortho_metrics(self._ortho())
        self.assertTrue(result["effective_measured"])
        self.assertGreater(result["effective_resolution_m"], 0.0)
        self.assertAlmostEqual(
            result["effective_resolution_m"],
            0.05 * result["effective_resolution_ratio"],
            places=12,
        )

    def test_an_untextured_mosaic_is_unmeasured_not_zero(self):
        flat = np.zeros((64, 64, 4), dtype=np.uint8)
        flat[:, :, 3] = 255
        ortho = metrics.Raster(flat, 0.0, 64.0, 0.05, 0.05)
        result = metrics.ortho_metrics(ortho)
        self.assertFalse(result["effective_measured"])
        self.assertIsNone(result["effective_resolution_ratio"])
        self.assertIsNone(result["effective_resolution_m"])

    def test_the_texture_window_is_taken_at_full_resolution(self):
        # A 3000 x 3000 mosaic with a 1024 cap must give a 1024-square window whose
        # pixels are still the mosaic's own cells (no stride).
        ortho = metrics.Raster(
            np.full((3000, 3000, 4), 200, dtype=np.uint8), 0.0, 3000.0, 0.1, 0.1
        )
        texture = metrics._texture_crop(ortho, np.ones((3000, 3000), dtype=bool), 1024)
        self.assertEqual(texture["samples"], 1024 * 1024)
        self.assertEqual(texture["window"][2:], [1024, 1024])


class DemMetricsTest(unittest.TestCase):
    def test_nodata_fraction_matches_the_sentinel_count(self):
        data = np.full((10, 10), 5.0)
        data[:2, :] = np.nan
        dem = metrics.Raster(data, 0.0, 10.0, 1.0, 1.0, nodata=np.nan)
        result = metrics.dem_metrics(dem)
        self.assertAlmostEqual(result["nodata_fraction"], 0.2)
        self.assertEqual(result["height_min_m"], 5.0)
        self.assertEqual(result["height_max_m"], 5.0)

    def test_a_reference_adds_a_difference_block(self):
        ours = metrics.Raster(np.full((8, 8), 2.0), 0.0, 8.0, 1.0, 1.0)
        reference = metrics.Raster(np.full((8, 8), 1.5), 0.0, 8.0, 1.0, 1.0)
        result = metrics.dem_metrics(ours, reference=reference)
        self.assertAlmostEqual(result["difference"]["median_m"], 0.5)


class CompareOrthoTest(unittest.TestCase):
    """The plan-view offset: colour-invariant, tile-robust, with a null path.

    Every mosaic here is built from one shared texture and shifted by an amount
    known by construction, so the recovered offset has an exact answer. The
    ``_scene`` helper is what makes "the same geometry, different colour" a fair
    test: both sides come from the same array, and only the gains differ.
    """

    @staticmethod
    def _scene(size=384, seed=11):
        """A 1/f texture: energy at every scale, like a real aerial scene.

        Adopted from the module's own conventions (the mask tests build a scene
        the same way) so the correlation has both fine edges and broad structure
        to lock onto.
        """
        rng = np.random.default_rng(seed)
        side = size + 64
        white = rng.random((side, side))
        spectrum = np.fft.fft2(white)
        ky = np.fft.fftfreq(side)[:, None]
        kx = np.fft.fftfreq(side)[None, :]
        frequency = np.hypot(kx, ky)
        frequency[0, 0] = 1.0
        field = np.real(np.fft.ifft2(spectrum / np.maximum(frequency, 1e-6) ** 0.8))
        field = (field - field.min()) / (field.max() - field.min())
        margin = 32
        return field[margin : margin + size, margin : margin + size]

    @staticmethod
    def _shift(img, dy, dx):
        """Band-limited shift of ``img`` (a Fourier phase ramp).

        Exact by construction, so a fractional shift is a fractional shift and
        not the interpolation error of a different resampler.
        """
        spectrum = np.fft.fft2(img)
        ky = np.fft.fftfreq(img.shape[0])[:, None]
        kx = np.fft.fftfreq(img.shape[1])[None, :]
        return np.real(np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (dx * kx + dy * ky))))

    def _offset(self, base, shifted, *, pixel=1.0):
        a = metrics.Raster(base, 0.0, float(base.shape[0]), pixel, pixel)
        b = metrics.Raster(shifted, 0.0, float(shifted.shape[0]), pixel, pixel)
        return metrics.compare_ortho_to_reference(a, b)

    def test_a_known_pixel_offset_becomes_metres(self):
        base = self._scene()
        shifted = self._shift(base, 0.0, 4.0)
        result = self._offset(base, shifted, pixel=0.5)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["dx_px"], 4.0, places=1)
        self.assertAlmostEqual(result["dx_m"], 2.0, delta=0.2)
        self.assertAlmostEqual(result["dy_px"], 0.0, places=1)
        self.assertTrue(result["offset_reliable"])

    def test_identical_rasters_have_zero_offset(self):
        base = self._scene()
        result = self._offset(base, base)
        self.assertTrue(result["offset_reliable"])
        self.assertAlmostEqual(result["offset_m"], 0.0, places=6)

    def test_a_tiny_overlap_is_reported_not_scored(self):
        a = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((4, 4)), 3.0, 4.0, 1.0, 1.0)
        result = metrics.compare_ortho_to_reference(a, b)
        self.assertFalse(result["comparable"])
        self.assertIn("reason", result)

    def test_diagonal_shifts_are_recovered_in_both_axes(self):
        base = self._scene()
        shifted = self._shift(base, 3.0, 4.0)
        result = self._offset(base, shifted)
        self.assertAlmostEqual(result["dy_px"], 3.0, delta=0.2)
        self.assertAlmostEqual(result["dx_px"], 4.0, delta=0.2)
        self.assertAlmostEqual(result["offset_m"], 5.0, delta=0.25)

    def test_the_legacy_value_is_still_reported(self):
        # One release of comparison: the old luminance number must still be in the
        # JSON beside the new one, so a reader can see what changed.
        base = self._scene()
        result = self._offset(base, self._shift(base, 0.0, 4.0))
        self.assertIn("offset_m_legacy", result)
        self.assertAlmostEqual(result["offset_m_legacy"], 4.0, delta=0.3)

    def test_the_metric_carries_its_definition_version(self):
        # The gate compares only equal versions, so the field itself is load
        # bearing: a missing version would silently compare a redefinition.
        result = self._offset(self._scene(), self._scene())
        self.assertEqual(result["metric_version"], metrics.OFFSET_METRIC_VERSION)

    def test_the_tile_statistics_are_reported(self):
        base = self._scene()
        result = self._offset(base, self._shift(base, 2.0, 2.0))
        tiles = result["tiles"]
        self.assertGreaterEqual(tiles["confident"], metrics.OFFSET_MIN_CONFIDENT_TILES)
        self.assertEqual(tiles["measured"], tiles["confident"])
        self.assertEqual(tiles["total"], 9)
        self.assertIsNotNone(result["offset_dispersion_m"])
        self.assertGreaterEqual(result["offset_dispersion_m"], 0.0)


class OffsetColourInvarianceTest(unittest.TestCase):
    """The offset must not move when only the COLOUR moves.

    This is the regression that motivated the metric redefinition: a per-camera
    gain change with identical geometry flipped the old luminance correlation
    from 0.985 m to 1.523 m on the aerial-nadir reference. The same scene is used
    on both sides here, so any movement is colour, not geometry.
    """

    @staticmethod
    def _scene(size=384, seed=11):
        return CompareOrthoTest._scene(size=size, seed=seed)

    @staticmethod
    def _shift(img, dy, dx):
        return CompareOrthoTest._shift(img, dy, dx)

    @staticmethod
    def _patchwork(img, gains):
        """Multiply each cell by the block gain of its region: a camera patchwork."""
        out = np.array(img, dtype=np.float64, copy=True)
        blocks_y, blocks_x = gains.shape
        h, w = out.shape
        for iy in range(blocks_y):
            r0, r1 = h * iy // blocks_y, h * (iy + 1) // blocks_y
            for ix in range(blocks_x):
                c0, c1 = w * ix // blocks_x, w * (ix + 1) // blocks_x
                out[r0:r1, c0:c1] *= gains[iy, ix]
        return out

    def _offset(self, ours, reference):
        a = metrics.Raster(ours, 0.0, float(ours.shape[0]), 1.0, 1.0)
        b = metrics.Raster(reference, 0.0, float(reference.shape[0]), 1.0, 1.0)
        return metrics.compare_ortho_to_reference(a, b)

    def test_a_per_region_gain_patchwork_does_not_move_the_offset(self):
        base = self._scene()
        shifted = self._shift(base, 0.0, 2.5)
        gains = np.array(
            [
                [1.12, 0.87, 1.15, 0.90],
                [0.85, 1.10, 0.88, 1.13],
                [1.16, 0.84, 1.09, 0.91],
                [0.86, 1.14, 0.90, 1.17],
            ]
        )
        plain = self._offset(base, shifted)
        coloured = self._offset(self._patchwork(base, gains), shifted)
        self.assertTrue(plain["offset_reliable"])
        self.assertTrue(coloured["offset_reliable"])
        self.assertAlmostEqual(coloured["dx_px"], plain["dx_px"], delta=0.2)
        self.assertAlmostEqual(coloured["offset_m"], plain["offset_m"], delta=0.2)

    def test_a_brightness_ramp_does_not_move_the_offset(self):
        # A stitching brightness ramp is the same failure mode as a per-camera
        # gain, spread across the mosaic instead of in blocks.
        base = self._scene()
        shifted = self._shift(base, 0.0, 2.5)
        ramp = np.linspace(0.83, 1.17, base.shape[1])[None, :]
        plain = self._offset(base, shifted)
        ramped = self._offset(base * ramp, shifted)
        self.assertAlmostEqual(ramped["dx_px"], plain["dx_px"], delta=0.2)
        self.assertAlmostEqual(ramped["dy_px"], plain["dy_px"], delta=0.2)

    def test_a_per_channel_gain_patchwork_does_not_move_the_offset(self):
        # Colour is three channels; a per-camera white-balance change is a gain
        # per CHANNEL. The luma weights keep a red/green cast visible, and the
        # local standardisation then removes it.
        base = self._scene()
        shifted = self._shift(base, 0.0, 2.5)
        gains = np.linspace(0.83, 1.17, 9).reshape(3, 3)
        ours = np.stack(
            [self._patchwork(base, gains * (0.90 + 0.10 * c)) for c in range(3)],
            axis=2,
        )
        theirs = np.stack([shifted, shifted, shifted], axis=2)
        plain = self._offset(base, shifted)
        coloured = self._offset(ours, theirs)
        self.assertAlmostEqual(coloured["dx_px"], plain["dx_px"], delta=0.2)
        self.assertAlmostEqual(coloured["offset_m"], plain["offset_m"], delta=0.2)

    def test_a_featureless_image_is_reported_unreliable(self):
        # A flat mosaic has no structure to register: the answer must be null and
        # flagged, never a confident-looking number the gate could act on.
        flat = np.full((384, 384), 0.5)
        result = self._offset(flat, flat)
        self.assertFalse(result["offset_reliable"])
        self.assertIsNone(result["offset_m"])
        self.assertIsNone(result["offset_dispersion_m"])
        self.assertIn("offset_reason", result)


class OffsetShiftAccuracyTest(unittest.TestCase):
    """Known shifts, including fractional ones, recovered to 0.2 px."""

    @staticmethod
    def _scene(size=384, seed=11):
        return CompareOrthoTest._scene(size=size, seed=seed)

    @staticmethod
    def _shift(img, dy, dx):
        return CompareOrthoTest._shift(img, dy, dx)

    def test_known_shifts_are_recovered_to_a_fifth_of_a_pixel(self):
        base = self._scene()
        for dy, dx in ((0.0, 2.5), (7.0, 0.0), (3.0, 4.0)):
            with self.subTest(dy=dy, dx=dx):
                result = metrics.compare_ortho_to_reference(
                    metrics.Raster(base, 0.0, 384.0, 1.0, 1.0),
                    metrics.Raster(self._shift(base, dy, dx), 0.0, 384.0, 1.0, 1.0),
                )
                self.assertTrue(result["offset_reliable"])
                self.assertAlmostEqual(result["dy_px"], dy, delta=0.2)
                self.assertAlmostEqual(result["dx_px"], dx, delta=0.2)

    def test_a_seven_pixel_shift_does_not_alias(self):
        # 7 px is inside the quarter-extent window but far enough that a wrong
        # spectrum-padding or a wrap-around bug shows up as 7 -> 384-7.
        base = self._scene()
        result = metrics.compare_ortho_to_reference(
            metrics.Raster(base, 0.0, 384.0, 1.0, 1.0),
            metrics.Raster(self._shift(base, 0.0, 7.0), 0.0, 384.0, 1.0, 1.0),
        )
        self.assertAlmostEqual(result["dx_px"], 7.0, delta=0.2)

    def test_unrelated_noise_has_no_confident_tile(self):
        # Two unrelated textures must not produce a confident offset: every tile's
        # peak is its own noise floor.
        rng = np.random.default_rng(1)
        first = metrics.Raster(rng.random((384, 384)), 0.0, 384.0, 1.0, 1.0)
        second = metrics.Raster(rng.random((384, 384)), 0.0, 384.0, 1.0, 1.0)
        result = metrics.compare_ortho_to_reference(first, second)
        self.assertFalse(result["offset_reliable"])
        self.assertIsNone(result["offset_m"])


class SeamStepTest(unittest.TestCase):
    """The reference-free colour-consistency metric, on mosaics built by hand.

    A seam is a flat scene rendered at two brightnesses, so the synthetic mosaics
    here are exactly that: a constant image with one column (or row) boundary
    where the value steps. Everything else is a control the metric must not fire
    on.
    """

    @staticmethod
    def _mosaic(value=0.5, size=256, step=None, *, axis="column", noise=0.0, seed=7):
        """A flat RGBA mosaic with an optional brightness step at one boundary.

        ``step`` is the signed change applied to every cell past ``boundary``.
        The value is in [0, 1] and written back as uint8, so the metric sees
        exactly what a GeoTIFF round trip would give it.
        """
        rng = np.random.default_rng(seed)
        rgb = np.full((size, size, 3), value, dtype=np.float64)
        if noise:
            rgb += rng.normal(0.0, noise, rgb.shape)
        if step is not None:
            if axis == "column":
                rgb[:, size // 2 :, :] += step
            else:
                rgb[size // 2 :, :, :] += step
        data = np.empty((size, size, 4), dtype=np.uint8)
        data[:, :, :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        data[:, :, 3] = 255
        return metrics.Raster(data, 0.0, float(size), 0.05, 0.05)

    def test_a_flat_mosaic_reports_zero_step(self):
        # Flat everywhere: every boundary line is flat and steps by nothing, so no
        # line is a seam and the result is a genuine 0.0 (measured, not omitted).
        result = metrics.ortho_colour_consistency(self._mosaic())
        self.assertTrue(result["seam_step_measured"])
        self.assertEqual(result["seam_lines"], 0)
        self.assertAlmostEqual(result["seam_step"], 0.0, places=6)
        self.assertAlmostEqual(result["seam_step_max"], 0.0, places=6)

    def test_a_column_step_is_measured_at_its_own_height(self):
        result = metrics.ortho_colour_consistency(self._mosaic(step=0.2))
        self.assertTrue(result["seam_step_measured"])
        self.assertEqual(result["seam_lines"], 1)
        # The step is 0.2 of full scale, up to the uint8 rounding.
        self.assertAlmostEqual(result["seam_step"], 0.2, delta=0.01)
        self.assertAlmostEqual(result["seam_step_max"], 0.2, delta=0.01)

    def test_a_row_step_is_measured_too(self):
        result = metrics.ortho_colour_consistency(self._mosaic(step=-0.15, axis="row"))
        self.assertTrue(result["seam_step_measured"])
        self.assertEqual(result["seam_lines"], 1)
        self.assertAlmostEqual(result["seam_step"], 0.15, delta=0.01)

    def test_correction_toward_the_mean_shrinks_the_step(self):
        # A 0.2 step corrected towards the mean leaves a 0.1 step: the measured
        # seam step must follow it down.
        before = metrics.ortho_colour_consistency(self._mosaic(step=0.2))
        after = metrics.ortho_colour_consistency(self._mosaic(step=0.1))
        self.assertLess(after["seam_step"], before["seam_step"])
        self.assertAlmostEqual(after["seam_step"], 0.1, delta=0.01)

    def test_moderate_noise_does_not_manufacture_a_seam(self):
        result = metrics.ortho_colour_consistency(self._mosaic(noise=0.02, seed=11))
        # Noise on a flat scene must not clear the seam floor: either no line is
        # promoted, or the reported step is below a visible one.
        if result["seam_step_measured"]:
            self.assertLess(result["seam_step"], metrics.SEAM_MIN_STEP)
        else:
            self.assertIsNone(result["seam_step"])

    def test_a_step_still_reads_through_moderate_noise(self):
        result = metrics.ortho_colour_consistency(
            self._mosaic(step=0.25, noise=0.01, seed=12)
        )
        self.assertTrue(result["seam_step_measured"])
        self.assertAlmostEqual(result["seam_step"], 0.25, delta=0.03)

    def test_holes_do_not_contribute_boundary_samples(self):
        mosaic = self._mosaic(step=0.2)
        # Knock out the right half's alpha over the top rows: those cells are
        # no-data, and the step must be measured only where both sides of the
        # boundary are valid. The surviving rows still carry the 0.2 step.
        mosaic.data[:64, 128:, 3] = 0
        result = metrics.ortho_colour_consistency(mosaic)
        self.assertTrue(result["seam_step_measured"])
        self.assertGreater(result["seam_step_p90"], 0.15)

    def test_an_all_nodata_mosaic_is_unmeasured(self):
        mosaic = self._mosaic()
        mosaic.data[:, :, 3] = 0
        result = metrics.ortho_colour_consistency(mosaic)
        self.assertFalse(result["seam_step_measured"])
        self.assertIsNone(result["seam_step"])

    def test_the_metric_is_reported_inside_ortho_metrics(self):
        result = metrics.ortho_metrics(self._mosaic(step=0.2))
        self.assertIn("colour_consistency", result)
        self.assertAlmostEqual(result["colour_consistency"]["seam_step"], 0.2, delta=0.01)


class ColourErrorVsReferenceTest(unittest.TestCase):
    """The reference-based residual after a global per-channel affine fit."""

    @staticmethod
    def _raster(rgb, *, size=None):
        rgb = np.asarray(rgb, dtype=np.float64)
        h, w = rgb.shape[:2]
        return metrics.Raster(rgb, 0.0, float(h), 1.0, 1.0)

    def test_identical_rasters_have_zero_residual(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, (64, 64, 3)).astype(np.float64)
        result = metrics.ortho_colour_error_vs_reference(self._raster(img), self._raster(img))
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["luma"]["median"], 0.0, places=9)

    def test_a_global_white_balance_difference_is_fitted_away(self):
        # Our mosaic is the reference scaled by 0.7: a pure global difference the
        # affine fit absorbs, so the residual is ~0 despite a large absolute gap.
        rng = np.random.default_rng(4)
        reference = rng.integers(40, 220, (64, 64, 3)).astype(np.float64)
        ours = reference * 0.7
        result = metrics.ortho_colour_error_vs_reference(self._raster(ours), self._raster(reference))
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["channels"]["red"]["slope"], 1.0 / 0.7, places=3)
        self.assertLess(result["luma"]["median"], 1e-3)

    def test_a_local_gain_error_survives_the_global_fit(self):
        # Left half correct, right half 25% too bright: no single affine can fix
        # both, so the residual is large on one side and the median moves.
        rng = np.random.default_rng(5)
        reference = rng.integers(40, 220, (64, 64, 3)).astype(np.float64)
        ours = reference.copy()
        ours[:, 32:, :] *= 1.25
        result = metrics.ortho_colour_error_vs_reference(self._raster(ours), self._raster(reference))
        self.assertTrue(result["comparable"])
        self.assertGreater(result["luma"]["median"], 0.01)
        self.assertGreater(result["luma"]["p90"], result["luma"]["median"])

    def test_a_tiny_overlap_is_reported_not_scored(self):
        a = self._raster(np.zeros((4, 4, 3)))
        b = metrics.Raster(np.zeros((4, 4, 3)), 3.0, 4.0, 1.0, 1.0)
        result = metrics.ortho_colour_error_vs_reference(a, b)
        self.assertFalse(result["comparable"])

    def test_a_flat_channel_is_not_a_division_by_zero(self):
        flat = np.full((16, 16, 3), 100.0)
        other = np.full((16, 16, 3), 140.0)
        result = metrics.ortho_colour_error_vs_reference(self._raster(flat), self._raster(other))
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["channels"]["red"]["slope"], 1.0)
        self.assertLess(result["luma"]["median"], 1e-6)


class EvaluateTest(unittest.TestCase):
    def test_missing_products_are_explicitly_skipped(self):
        spec = metrics.DatasetSpec(name="synthetic", images=[])
        result = metrics.evaluate(spec, ortho_path=None, dem_path=None, cloud_path=None)
        self.assertIn("skipped", result["reference_free"]["ortho"])
        self.assertIn("skipped", result["reference_free"]["dem"])
        self.assertIn("skipped", result["reference_free"]["cameras"])
        self.assertIn("skipped", result["reference_based"]["ortho"])
        self.assertIn("skipped", result["reference_based"]["dem"])
        self.assertIn("skipped", result["reference_based"]["cloud"])

    def test_an_unreadable_product_is_reported_not_raised(self):
        spec = metrics.DatasetSpec(name="synthetic", images=[])
        result = metrics.evaluate(
            spec, ortho_path="/nonexistent/ortho.tif", dem_path=None, cloud_path=None
        )
        self.assertIn("skipped", result["reference_free"]["ortho"])

    def test_camera_stats_are_passed_through(self):
        spec = metrics.DatasetSpec(name="synthetic", images=[])
        result = metrics.evaluate(
            spec, ortho_path=None, dem_path=None, cloud_path=None,
            camera_stats={"rms_px": 0.4, "pairs": 100},
        )
        self.assertEqual(result["reference_free"]["cameras"]["rms_px"], 0.4)

    def test_the_licence_travels_with_the_result(self):
        spec = metrics.DatasetSpec(
            name="synthetic", images=[], licence={"spdx": "CC-BY-4.0", "url": "https://example"}
        )
        result = metrics.evaluate(spec, ortho_path=None, dem_path=None, cloud_path=None)
        self.assertEqual(result["licence"]["spdx"], "CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()
