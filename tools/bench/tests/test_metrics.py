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
        result = metrics.mtf50_effective_gsd(image, pixel_size_m=0.02)
        self.assertTrue(result["textured"])
        # White noise has almost no spatial correlation, so any blur halves its
        # gradient energy immediately: the ratio stays under two.
        self.assertLess(result["ratio"], 2.0)
        self.assertGreater(result["ratio"], 0.0)

    def test_a_pre_blurred_image_reports_a_larger_ratio(self):
        rng = np.random.default_rng(11)
        image = rng.random((128, 128))
        sharp = metrics.mtf50_effective_gsd(image, pixel_size_m=0.02)
        soft = metrics.mtf50_effective_gsd(metrics._box_blur(image, 2), pixel_size_m=0.02)
        self.assertGreater(soft["ratio"], sharp["ratio"])

    def test_effective_gsd_is_the_cell_size_times_the_ratio(self):
        rng = np.random.default_rng(3)
        image = rng.random((64, 64))
        one = metrics.mtf50_effective_gsd(image, pixel_size_m=0.01)
        two = metrics.mtf50_effective_gsd(image, pixel_size_m=0.02)
        # The ratio is a property of the image alone; only the metre conversion
        # follows the cell size.
        self.assertAlmostEqual(two["ratio"], one["ratio"], places=12)
        self.assertAlmostEqual(two["effective_gsd_m"], 2.0 * one["effective_gsd_m"], places=12)

    def test_a_flat_image_is_not_textured(self):
        result = metrics.mtf50_effective_gsd(np.full((32, 32), 0.25), pixel_size_m=0.02)
        self.assertFalse(result["textured"])
        self.assertFalse(result["measured"])
        self.assertIsNone(result["ratio"])

    def test_a_search_that_never_halves_is_unmeasured_not_capped(self):
        # A very smooth image whose energy only falls a little under the whole
        # search: the answer is "no half-energy point inside the cap", never the
        # cap itself dressed up as a measurement.
        image = np.tile(np.arange(64, dtype=np.float64), (64, 1))
        result = metrics.mtf50_effective_gsd(image, pixel_size_m=0.05, max_blur_radius=2)
        self.assertFalse(result["measured"])
        self.assertIsNone(result["ratio"])
        self.assertIsNone(result["effective_gsd_m"])
        self.assertIn("cap", result["reason"])

    def test_a_non_positive_pixel_size_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.mtf50_effective_gsd(np.ones((8, 8)), pixel_size_m=0.0)


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
        # effective_gsd_m = 0.05 * ratio, whatever the imagery's own GSD was.
        result = metrics.ortho_metrics(self._ortho())
        self.assertTrue(result["effective_measured"])
        self.assertGreater(result["effective_gsd_m"], 0.0)
        self.assertAlmostEqual(
            result["effective_gsd_m"],
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
        self.assertIsNone(result["effective_gsd_m"])

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
    def test_a_known_pixel_offset_becomes_metres(self):
        rng = np.random.default_rng(21)
        size = 96
        base = metrics._box_blur(rng.random((size, size)), 1)
        shifted = np.roll(base, (0, 4), axis=(0, 1))
        a = metrics.Raster(base, 0.0, float(size), 0.5, 0.5)
        b = metrics.Raster(shifted, 0.0, float(size), 0.5, 0.5)
        result = metrics.compare_ortho_to_reference(a, b)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["dx_px"], 4.0, places=1)
        self.assertAlmostEqual(result["dx_m"], 2.0, delta=0.2)
        self.assertAlmostEqual(result["dy_px"], 0.0, places=1)

    def test_identical_rasters_have_zero_offset(self):
        rng = np.random.default_rng(22)
        data = metrics._box_blur(rng.random((64, 64)), 1)
        raster = metrics.Raster(data, 0.0, 64.0, 1.0, 1.0)
        result = metrics.compare_ortho_to_reference(raster, raster)
        self.assertAlmostEqual(result["offset_m"], 0.0, places=6)

    def test_a_tiny_overlap_is_reported_not_scored(self):
        a = metrics.Raster(np.zeros((4, 4)), 0.0, 4.0, 1.0, 1.0)
        b = metrics.Raster(np.zeros((4, 4)), 3.0, 4.0, 1.0, 1.0)
        result = metrics.compare_ortho_to_reference(a, b)
        self.assertFalse(result["comparable"])
        self.assertIn("reason", result)


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
