#!/usr/bin/env python3
"""Unit tests for the EuroSDR observation measurement script.

The engine driver needs the built ``alphashape`` module and the multi-GB
archive, so it is not tested here. What is tested is everything that decides
WHICH observations are written, because that is where a silent mistake is most
expensive: a projection that refined onto the wrong target, a station read from
the wrong xlsx column, or a gcp_list row with the wrong pixel convention would
all produce a plausible-looking file that quietly poisons the ground-control
validation.

Run with plain unittest (no pytest):

    python3 -m unittest discover -s tools/bench/tests -v
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eurosdr_observations as eo  # noqa: E402


class GeodesyTest(unittest.TestCase):
    """The projection geometry must match the engine's own conventions."""

    def test_geodetic_to_ecef_matches_known_point(self):
        # The WGS84 origin: lon 0, lat 0, height 0 is (a, 0, 0).
        x, y, z = eo.geodetic_to_ecef(0.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 6378137.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 0.0, places=6)

    def test_geodetic_to_ecef_north_pole(self):
        # At the pole the equatorial radius drops by the polar radius b.
        x, y, z = eo.geodetic_to_ecef(0.0, 90.0, 0.0)
        b = 6378137.0 * (1.0 - 1.0 / 298.257223563)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, b, places=3)

    def test_ecef_to_local_round_trips_an_identity_transform(self):
        # An identity transform leaves the ECEF point unchanged.
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        point = (3640741.0, -119474.0, 5218310.0)
        local = eo.ecef_to_local(point, identity)
        for got, want in zip(local, point):
            self.assertAlmostEqual(got, want, places=6)

    def test_ecef_to_local_inverts_a_rigid_rotation(self):
        # A georeferenced chunk's transform is rigid (scale 1): a 90-degree yaw
        # about z plus a translation. The rotation block is [[0,-1,0],[1,0,0],
        # [0,0,1]], so local (1, 0, 0) maps to (0, 1, 0) plus the shift
        # (10, 20, 30) = (10, 21, 30). The inverse must recover (1, 0, 0).
        transform = [[0.0, -1.0, 0.0, 10.0],
                     [1.0, 0.0, 0.0, 20.0],
                     [0.0, 0.0, 1.0, 30.0],
                     [0.0, 0.0, 0.0, 1.0]]
        ecef = (10.0, 21.0, 30.0)  # local (1, 0, 0)
        local = eo.ecef_to_local(ecef, transform)
        self.assertAlmostEqual(local[0], 1.0, places=9)
        self.assertAlmostEqual(local[1], 0.0, places=9)
        self.assertAlmostEqual(local[2], 0.0, places=9)

    def test_ecef_to_local_rejects_a_singular_transform(self):
        singular = [[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        with self.assertRaises(ValueError):
            eo.ecef_to_local((1.0, 2.0, 3.0), singular)


class ProjectionTest(unittest.TestCase):
    """The forward projection must match the engine's pixel convention."""

    def calibration(self, **overrides):
        base = {
            "type": "frame", "width": 5472, "height": 3648, "f": 3666.0,
            "cx": 0.0, "cy": 0.0, "b1": 0.0, "b2": 0.0,
            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
            "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0,
        }
        base.update(overrides)
        return base

    def test_point_on_the_optical_axis_lands_at_the_image_centre(self):
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        pixel = eo.project_point((0.0, 0.0, 10.0), identity, self.calibration())
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], 5472 / 2.0, places=6)
        self.assertAlmostEqual(pixel[1], 3648 / 2.0, places=6)

    def test_point_behind_the_camera_has_no_projection(self):
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        self.assertIsNone(eo.project_point((0.0, 0.0, -10.0), identity,
                                           self.calibration()))

    def test_point_outside_the_image_has_no_projection(self):
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        # Far to the right of the frame.
        self.assertIsNone(eo.project_point((1000.0, 0.0, 1.0), identity,
                                           self.calibration()))

    def test_principal_point_offset_shifts_the_projection(self):
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        pixel = eo.project_point((0.0, 0.0, 10.0), identity,
                                 self.calibration(cx=10.0, cy=-5.0))
        self.assertAlmostEqual(pixel[0], 5472 / 2.0 + 10.0, places=6)
        self.assertAlmostEqual(pixel[1], 3648 / 2.0 - 5.0, places=6)

    def test_fisheye_is_refused(self):
        identity = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
        with self.assertRaises(ValueError):
            eo.project_point((0.0, 0.0, 10.0), identity,
                             self.calibration(type="fisheye"))

    def test_apparent_radius_scales_with_focal_over_depth(self):
        calibration = self.calibration(f=3666.0)
        # 300 mm target at 100 m: radius 0.15 m, so 0.15 * 3666 / 100 px.
        radius = eo.apparent_target_radius_px(calibration, 100.0)
        self.assertAlmostEqual(radius, 0.15 * 3666.0 / 100.0, places=6)

    def test_apparent_radius_is_zero_without_scale(self):
        self.assertEqual(eo.apparent_target_radius_px(self.calibration(), 0.0), 0.0)


class AcceptanceTest(unittest.TestCase):
    """The acceptance gate is what decides which observations are written."""

    def test_a_detection_on_the_target_is_accepted(self):
        # A target found 2 px from the projection, on a 20 px target radius,
        # with a 1 px residual.
        self.assertTrue(eo.accept_observation(2.0, 20.0, 1.0))

    def test_a_detection_at_the_projection_is_accepted(self):
        # A detected target is already on the target, so a detection exactly at
        # the projection (shift 0) is a valid observation.
        self.assertTrue(eo.accept_observation(0.0, 20.0, 0.0))

    def test_a_detection_beyond_one_target_radius_is_rejected(self):
        # More than a target radius away is more likely a neighbouring target.
        # This is the target-relative bound, which applies only when no search
        # radius is supplied (a caller that projects accurately).
        self.assertFalse(eo.accept_observation(25.0, 20.0, 1.0))

    def test_a_search_radius_supersedes_the_target_relative_bound(self):
        # The RTK-only projection can sit ~16 px from the printed centre, which
        # is more than one target radius (~10 px) but well inside the search
        # radius. The search radius is the "right target" gate in that case: the
        # neighbouring target is thousands of pixels away, so a detection inside
        # the radius is unambiguously this station.
        self.assertTrue(eo.accept_observation(16.0, 10.0, None,
                                              search_radius_px=40.0))
        # Beyond the search radius it is still rejected.
        self.assertFalse(eo.accept_observation(45.0, 10.0, None,
                                               search_radius_px=40.0))

    def test_a_detection_beyond_the_search_radius_is_rejected(self):
        self.assertFalse(eo.accept_observation(50.0, 20.0, 1.0,
                                               search_radius_px=40.0))

    def test_a_large_residual_is_rejected(self):
        # The detection is a different target: the triangulated position
        # reprojects far from the detected pixel.
        self.assertFalse(eo.accept_observation(2.0, 20.0, 50.0))

    def test_a_scale_less_view_uses_the_search_radius(self):
        # With no target scale, the search radius is the only bound.
        self.assertTrue(eo.accept_observation(3.0, 0.0, None,
                                              search_radius_px=40.0))
        self.assertFalse(eo.accept_observation(50.0, 0.0, None,
                                               search_radius_px=40.0))

    def test_a_non_finite_shift_is_rejected(self):
        self.assertFalse(eo.accept_observation(float("nan"), 20.0, 1.0))
        self.assertFalse(eo.accept_observation(float("inf"), 20.0, 1.0))

    def test_a_non_finite_residual_is_rejected(self):
        self.assertFalse(eo.accept_observation(2.0, 20.0, float("nan")))


class GcpListTest(unittest.TestCase):
    """The gcp_list writer must produce the format the bench parses."""

    def stations(self):
        return [
            {"label": "GA1", "lat": 55.263526429, "lon": -1.874199729,
             "height": 211.0931, "role": "gcp"},
            {"label": "A2", "lat": 55.263009727, "lon": -1.879550263,
             "height": 228.9753, "role": "check_point"},
        ]

    def test_header_is_epsg_4979(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_list.txt"
            eo.write_gcp_list(path, self.stations(), {})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "EPSG:4979")

    def test_one_row_per_observation_with_the_label_last(self):
        observations = {
            "GA1": [{"image": "100_0004_0001", "px": 1234.5, "py": 987.25}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_list.txt"
            rows = eo.write_gcp_list(path, self.stations(), observations)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows, 1)
        self.assertEqual(len(lines), 2)
        tokens = lines[1].split()
        self.assertEqual(len(tokens), 7)
        # lon lat height px py image label
        self.assertAlmostEqual(float(tokens[0]), -1.874199729, places=9)
        self.assertAlmostEqual(float(tokens[1]), 55.263526429, places=9)
        self.assertAlmostEqual(float(tokens[2]), 211.0931, places=4)
        self.assertAlmostEqual(float(tokens[3]), 1234.5, places=3)
        self.assertAlmostEqual(float(tokens[4]), 987.25, places=3)
        self.assertEqual(tokens[5], "100_0004_0001")
        self.assertEqual(tokens[6], "GA1")

    def test_a_station_with_no_observation_writes_no_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_list.txt"
            rows = eo.write_gcp_list(path, self.stations(), {})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows, 0)
        self.assertEqual(lines, ["EPSG:4979"])

    def test_rows_are_sorted_by_image_within_a_station(self):
        observations = {
            "GA1": [
                {"image": "100_0004_0003", "px": 1.0, "py": 2.0},
                {"image": "100_0004_0001", "px": 3.0, "py": 4.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_list.txt"
            eo.write_gcp_list(path, self.stations(), observations)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[1].split()[5], "100_0004_0001")
        self.assertEqual(lines[2].split()[5], "100_0004_0003")


class SummaryTest(unittest.TestCase):
    """The summary must count views and acceptance the way the report reads it."""

    def stations(self):
        return [
            {"label": "GA1", "lat": 0.0, "lon": 0.0, "height": 0.0, "role": "gcp"},
            {"label": "A2", "lat": 0.0, "lon": 0.0, "height": 0.0, "role": "check_point"},
        ]

    def test_counts_and_acceptance_rate(self):
        per_point = {
            "GA1": [
                {"image": "a", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
                {"image": "b", "shift_px": 0.0, "residual_px": None, "accepted": False},
                {"image": "c", "shift_px": 1.5, "residual_px": 0.5, "accepted": True},
            ],
            "A2": [
                {"image": "a", "shift_px": 2.0, "residual_px": 1.0, "accepted": True},
            ],
        }
        summary = eo.summarise(self.stations(), {}, per_point)
        self.assertEqual(summary["stations"], 2)
        self.assertEqual(summary["gcps"], 1)
        self.assertEqual(summary["check_points"], 1)
        self.assertEqual(summary["views_projected"], 4)
        self.assertEqual(summary["views_accepted"], 3)
        self.assertAlmostEqual(summary["acceptance_rate"], 3.0 / 4.0, places=9)
        # GA1 has two accepted views, which meets MIN_VIEWS_PER_POINT.
        self.assertEqual(summary["gcps_with_min_views"], 1)

    def test_a_station_with_no_views_is_listed_with_zero(self):
        summary = eo.summarise(self.stations(), {}, {})
        self.assertEqual(summary["views_projected"], 0)
        self.assertIsNone(summary["acceptance_rate"])
        self.assertEqual(summary["points"][0]["views_accepted"], 0)
        self.assertEqual(summary["points"][0]["images"], [])

    def test_mean_and_max_shift_use_accepted_views_only(self):
        per_point = {
            "GA1": [
                {"image": "a", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
                {"image": "b", "shift_px": 3.0, "residual_px": 0.5, "accepted": True},
                {"image": "c", "shift_px": 99.0, "residual_px": 0.5, "accepted": False},
            ],
        }
        summary = eo.summarise(self.stations(), {}, per_point)
        point = summary["points"][0]
        self.assertAlmostEqual(point["mean_shift_px"], 2.0, places=9)
        self.assertAlmostEqual(point["max_shift_px"], 3.0, places=9)


class SelectImagesTest(unittest.TestCase):
    """The strided subset must keep the whole site covered, not just the middle."""

    def test_no_limit_keeps_everything(self):
        images = [f"img{i:04d}.JPG" for i in range(430)]
        selected, stride = eo.select_images(images, None)
        self.assertEqual(selected, images)
        self.assertEqual(stride, 1)

    def test_a_limit_at_or_above_the_count_keeps_everything(self):
        images = [f"img{i:04d}.JPG" for i in range(10)]
        selected, stride = eo.select_images(images, 10)
        self.assertEqual(selected, images)
        self.assertEqual(stride, 1)

    def test_a_stride_keeps_the_first_and_last_strips(self):
        # 430 images, limit 215 -> stride 2, so every other frame. The first and
        # last frames are kept, which is what keeps the site's extent.
        images = [f"img{i:04d}.JPG" for i in range(430)]
        selected, stride = eo.select_images(images, 215)
        self.assertEqual(stride, 2)
        self.assertEqual(len(selected), 215)
        self.assertEqual(selected[0], images[0])
        self.assertEqual(selected[-1], images[-2])

    def test_the_stride_is_the_ceiling_of_the_ratio(self):
        images = [f"img{i:04d}.JPG" for i in range(430)]
        selected, stride = eo.select_images(images, 100)
        self.assertEqual(stride, 5)  # ceil(430 / 100) = 5
        self.assertLessEqual(len(selected), 100)

    def test_a_zero_limit_keeps_everything(self):
        images = [f"img{i:04d}.JPG" for i in range(10)]
        selected, stride = eo.select_images(images, 0)
        self.assertEqual(selected, images)
        self.assertEqual(stride, 1)


class DefaultsTest(unittest.TestCase):
    """The detection defaults must suit the printed target, not a coded ring."""

    def test_detection_runs_at_full_resolution(self):
        # The 100 mm black centre is ~3.3 px radius at the 1.5 cm GSD. The auto
        # downscale of 2 halves it below the detector's 3 px floor, so the
        # default must force full resolution.
        self.assertEqual(eo.DEFAULT_MARKER_DOWNSCALE, 1)

    def test_the_minimum_target_admits_the_black_centre(self):
        # The black centre is ~3.3 px radius; a floor above that rejects it.
        self.assertLessEqual(eo.DEFAULT_MINIMUM_TARGET_PX, 3.0)

    def test_the_target_is_the_plain_circle(self):
        # The publisher printed plain 300 mm disks, not coded rings.
        self.assertEqual(eo.DEFAULT_TARGET_TYPE, "Circular")


class SelectRefineViewsTest(unittest.TestCase):
    """Each marker is refined against its closest observers, not all of them."""

    def test_keeps_the_largest_apparent_targets(self):
        projections = {
            "GA1": [
                {"camera": "a", "px": 0.0, "py": 0.0, "target_radius_px": 5.0},
                {"camera": "b", "px": 0.0, "py": 0.0, "target_radius_px": 20.0},
                {"camera": "c", "px": 0.0, "py": 0.0, "target_radius_px": 12.0},
            ],
        }
        selected = eo.select_refine_views(projections, 2)
        cameras = [view["camera"] for view in selected["GA1"]]
        self.assertEqual(cameras, ["b", "c"])

    def test_a_marker_with_fewer_views_keeps_them_all(self):
        projections = {
            "GA1": [{"camera": "a", "px": 0.0, "py": 0.0, "target_radius_px": 5.0}],
        }
        selected = eo.select_refine_views(projections, 16)
        self.assertEqual(len(selected["GA1"]), 1)

    def test_every_station_is_present(self):
        projections = {
            "GA1": [{"camera": "a", "px": 0.0, "py": 0.0, "target_radius_px": 5.0}],
            "A2": [{"camera": "b", "px": 0.0, "py": 0.0, "target_radius_px": 6.0}],
        }
        selected = eo.select_refine_views(projections, 16)
        self.assertEqual(set(selected), {"GA1", "A2"})


class XlsxReaderTest(unittest.TestCase):
    """The xlsx reader must take the ETRS89 ellipsoidal block, not OSGB36/ODN."""

    def setUp(self):
        # openpyxl is a benchmark-only dependency: the measurement workflow
        # installs it, but the Plan job's unit-test step runs on the stdlib +
        # numpy alone. Skip rather than error, exactly as the rasterio tests in
        # test_metrics.py do, so a missing optional reader never fails the plan.
        try:
            import openpyxl  # noqa: F401, PLC0415
        except ImportError:  # pragma: no cover - openpyxl is in the workflow
            self.skipTest("openpyxl is not installed")

    def write_workbook(self, directory, rows):
        import openpyxl  # noqa: PLC0415 - test-only dependency

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = eo.XLSX_SHEET
        # Two header rows, then the data rows the caller supplies.
        sheet.append(["Station", "ETRS89 Cartesian", None, None,
                      "ETRS89 Geodetic (GRS80 ellipsoid)", None, None,
                      "OSGB36/ODN", None, None])
        sheet.append(["A1", "X_ave", "Y_ave", "Z_ave",
                      "lat_ave", "lon_ave", "h_ave", "E_ave", "N_ave", "H_ave"])
        for row in rows:
            sheet.append(row)
        path = Path(directory) / "coords.xlsx"
        workbook.save(path)
        return path

    def test_reads_the_etrs89_geodetic_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ["GA1", 3640000.0, -119000.0, 5218000.0,
                 55.263526429, -1.874199729, 211.0931,
                 407000.0, 596000.0, 161.0],
            ])
            stations = eo.read_stations(path)
        self.assertEqual(len(stations), 1)
        station = stations[0]
        self.assertEqual(station["label"], "GA1")
        self.assertAlmostEqual(station["lat"], 55.263526429, places=9)
        self.assertAlmostEqual(station["lon"], -1.874199729, places=9)
        # The ellipsoidal height, not the orthometric 161.0.
        self.assertAlmostEqual(station["height"], 211.0931, places=4)
        self.assertEqual(station["role"], "gcp")

    def test_a_non_gcp_station_is_a_check_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ["A2", 0.0, 0.0, 0.0, 55.263009727, -1.879550263, 228.9753,
                 0.0, 0.0, 178.0],
            ])
            stations = eo.read_stations(path)
        self.assertEqual(stations[0]["role"], "check_point")

    def test_an_unnamed_station_gets_the_unnamed_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                [None, 0.0, 0.0, 0.0, 55.262911463, -1.876660186, 222.2881,
                 0.0, 0.0, 171.0],
            ])
            stations = eo.read_stations(path)
        self.assertEqual(stations[0]["label"], eo.UNNAMED_STATION_LABEL)
        self.assertEqual(stations[0]["role"], "check_point")

    def test_a_partial_coordinate_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ["A2", 0.0, 0.0, 0.0, 55.263, None, 228.9, 0.0, 0.0, 178.0],
            ])
            with self.assertRaises(ValueError):
                eo.read_stations(path)

    def test_a_non_finite_coordinate_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ["A2", 0.0, 0.0, 0.0, float("nan"), -1.879, 228.9, 0.0, 0.0, 178.0],
            ])
            with self.assertRaises(ValueError):
                eo.read_stations(path)

    def test_a_missing_sheet_raises(self):
        import openpyxl  # noqa: PLC0415 - test-only dependency

        with tempfile.TemporaryDirectory() as directory:
            workbook = openpyxl.Workbook()
            workbook.active.title = "NotTheSheet"
            path = Path(directory) / "coords.xlsx"
            workbook.save(path)
            with self.assertRaises(ValueError):
                eo.read_stations(path)


if __name__ == "__main__":
    unittest.main()
