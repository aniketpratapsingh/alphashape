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


class GcpNamesTest(unittest.TestCase):
    """The survey CSV must carry the labels and ellipsoidal heights the
    protocol matches the gcp_list against."""

    def stations(self):
        return [
            {"label": "GA1", "lat": 55.263526429, "lon": -1.874199729,
             "height": 211.0931, "role": "gcp"},
            {"label": "A2", "lat": 55.263009727, "lon": -1.879550263,
             "height": 228.9753, "role": "check_point"},
        ]

    def test_header_is_the_publisher_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_names.csv"
            eo.write_gcp_names(path, self.stations())
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            lines[0], "Name;Easting;Northing;Elevation;Longitude;Latitude")

    def test_one_row_per_station_with_the_ellipsoidal_height(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_names.csv"
            rows = eo.write_gcp_names(path, self.stations())
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows, 2)
        fields = lines[1].split(";")
        self.assertEqual(fields[0], "GA1")
        self.assertAlmostEqual(float(fields[3]), 211.0931, places=4)
        self.assertAlmostEqual(float(fields[4]), -1.874199729, places=9)
        self.assertAlmostEqual(float(fields[5]), 55.263526429, places=9)

    def test_the_protocol_parser_reads_it_back(self):
        # The CSV must round-trip through the protocol's own parser, so a
        # format drift is caught here rather than in a multi-hour bench run.
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
            "engine", "tests"))
        try:
            import run_gcp_validation as rgv  # noqa: PLC0415
        except ImportError:  # pragma: no cover - engine tree always present
            self.skipTest("run_gcp_validation is not importable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp_names.csv"
            eo.write_gcp_names(path, self.stations())
            columns, rows = rgv.parse_gcp_names_csv(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "GA1")
        self.assertAlmostEqual(rows[0]["elevation"], 211.0931, places=4)
        self.assertAlmostEqual(rows[0]["longitude"], -1.874199729, places=9)
        self.assertAlmostEqual(rows[0]["latitude"], 55.263526429, places=9)


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

    def test_without_geometry_every_station_is_scored(self):
        # The exclusion rule needs the camera centres; without them every
        # station is reported as scored rather than silently dropped.
        summary = eo.summarise(self.stations(), {}, {})
        self.assertTrue(all(point["scored"] for point in summary["points"]))
        self.assertEqual(summary["excluded_stations"], [])
        self.assertEqual(summary["scored_stations"], 2)

    def test_a_station_with_too_few_images_is_excluded_with_a_reason(self):
        per_point = {
            "GA1": [
                {"image": "a", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
                {"image": "b", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
            ],
        }
        centres = {"a": (0.0, 0.0, 0.0), "b": (10.0, 0.0, 0.0)}
        local = {"GA1": (5.0, 0.0, 50.0), "A2": (0.0, 0.0, 50.0)}
        summary = eo.summarise(self.stations(), {}, per_point,
                               centres=centres, local=local)
        ga1 = next(point for point in summary["points"] if point["label"] == "GA1")
        self.assertFalse(ga1["scored"])
        self.assertIn("accepted image", ga1["exclusion_reason"])
        excluded = {record["label"]: record for record in summary["excluded_stations"]}
        self.assertIn("GA1", excluded)
        self.assertIn("reason", excluded["GA1"])
        # A2 has no accepted images at all, so it is excluded too.
        self.assertIn("A2", excluded)
        self.assertEqual(summary["scored_stations"], 0)

    def test_a_well_conditioned_station_is_scored(self):
        per_point = {
            "GA1": [
                {"image": "a", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
                {"image": "b", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
                {"image": "c", "shift_px": 1.0, "residual_px": 0.5, "accepted": True},
            ],
        }
        centres = {"a": (0.0, 0.0, 0.0), "b": (10.0, 0.0, 0.0), "c": (5.0, 10.0, 0.0)}
        local = {"GA1": (5.0, 5.0, 50.0), "A2": (0.0, 0.0, 50.0)}
        summary = eo.summarise(self.stations(), {}, per_point,
                               centres=centres, local=local)
        ga1 = next(point for point in summary["points"] if point["label"] == "GA1")
        self.assertTrue(ga1["scored"])
        self.assertIsNone(ga1["exclusion_reason"])


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

    def test_both_polarities_run_bright_first(self):
        # The targets are bright disks with a small black centre. The bright
        # (inverted) pass finds the robust ~10 px disk and must run first so its
        # better-centred detection wins a camera; the dark pass fills in.
        self.assertEqual(eo.DEFAULT_INVERTED_PASSES, (True, False))

    def test_the_search_radius_covers_the_rtk_error(self):
        # The RTK-only georeference is ~16 px off at 1.5 cm GSD; the search
        # radius must cover several sigma of that, not one target radius.
        self.assertGreaterEqual(eo.DEFAULT_SEARCH_RADIUS_PX, 40.0)

    def test_the_block_is_bundle_adjusted_before_projecting(self):
        # AlignCameras georeferences with ONE global similarity, and on this
        # flight that fit keeps only 6 of 430 cameras inside its inlier
        # threshold: the block drifts, the projections land tens of pixels off
        # the targets, and the multi-view consensus fails. The bundle adjustment
        # with the camera GNSS priors is what removes the drift, so it must be
        # on by default.
        self.assertTrue(eo.DEFAULT_OPTIMIZE)


class AlignBlockTest(unittest.TestCase):
    """The alignment places the cameras from RTK before any station exists."""

    def _fake_alphashape(self, calls, cameras=()):
        class FakeCamera:
            def __init__(self, enabled):
                self.enabled = enabled

        class FakeChunk:
            def __init__(self):
                self.crs_epsg = None
                self.cameras = [FakeCamera(enabled) for enabled in cameras]
                self.meta = {}

            def optimizeCameras(self, **kwargs):
                calls.append(("optimizeCameras", kwargs))

        class FakeDocument:
            def __init__(self):
                self.active_chunk = None

            def add_chunk(self, name):
                chunk = FakeChunk()
                self.active_chunk = chunk
                return chunk

            def run_task(self, name, **kwargs):
                calls.append((name, kwargs))

        class FakeModule:
            Document = FakeDocument

        return FakeModule()

    def test_the_cameras_are_placed_from_their_reference(self):
        # AlignCameras georeferences with ONE global similarity, and on this
        # flight that fit keeps only 6 of 430 cameras inside its inlier
        # threshold: the similarity is extrapolated from a small cluster and is
        # metres off across the block. AlignFromReference places every camera
        # from its own RTK position and attitude instead, so the global fit is
        # not used at all.
        calls = []
        eo.align_block(self._fake_alphashape(calls), ["a.jpg", "b.jpg", "c.jpg"], 250)
        names = [name for name, _ in calls]
        self.assertIn("AlignFromReference", names)
        self.assertNotIn("AlignCameras", names)

    def test_the_tie_points_are_rebuilt_before_the_bundle_adjustment(self):
        # AlignFromReference clears the tie points (they belonged to the old
        # frame), so they must be triangulated from the placed poses before the
        # bundle adjustment can use them.
        calls = []
        eo.align_block(self._fake_alphashape(calls), ["a.jpg", "b.jpg", "c.jpg"], 250)
        names = [name for name, _ in calls]
        self.assertIn("TriangulateTiePoints", names)
        self.assertIn("optimizeCameras", names)
        self.assertGreater(names.index("TriangulateTiePoints"),
                           names.index("AlignFromReference"))
        self.assertGreater(names.index("optimizeCameras"),
                           names.index("TriangulateTiePoints"))

    def test_the_gimbal_attitude_is_imported(self):
        # AlignFromReference cannot place a camera without an attitude, so the
        # import must read the XMP gimbal angles.
        calls = []
        eo.align_block(self._fake_alphashape(calls), ["a.jpg", "b.jpg", "c.jpg"], 250)
        add_photos = [kwargs for name, kwargs in calls if name == "AddPhotos"]
        self.assertTrue(add_photos)
        self.assertTrue(add_photos[0].get("load_xmp_orientation"))

    def test_every_camera_is_enabled_before_it_is_placed(self):
        # AddPhotos leaves every camera disabled, and AlignFromReference only
        # places ENABLED cameras: without enabling them first the task would
        # place none and the block would stay unaligned.
        calls = []
        module = self._fake_alphashape(calls, cameras=(False, False, False))
        document, chunk = eo.align_block(module, ["a.jpg", "b.jpg", "c.jpg"], 250)
        self.assertTrue(all(camera.enabled for camera in chunk.cameras))

    def test_no_station_is_added_before_the_bundle_adjustment(self):
        # The measurement must not let the surveyed control influence the
        # alignment that is used to project it: the alignment runs on a chunk
        # with no markers at all.
        calls = []
        eo.align_block(self._fake_alphashape(calls), ["a.jpg", "b.jpg", "c.jpg"], 250)
        self.assertNotIn("addMarker", [name for name, _ in calls])

    def test_optimize_can_be_switched_off(self):
        # The comparison path is the plain incremental alignment.
        calls = []
        eo.align_block(self._fake_alphashape(calls), ["a.jpg", "b.jpg", "c.jpg"], 250,
                       optimize=False)
        names = [name for name, _ in calls]
        self.assertIn("AlignCameras", names)
        self.assertNotIn("AlignFromReference", names)
        self.assertNotIn("optimizeCameras", names)


class CameraReferenceStatsTest(unittest.TestCase):
    """The summary must say whether the cameras carry the RTK solution."""

    class _Camera:
        def __init__(self, enabled=True, reference=True, flag=50, accuracy=(0.02, 0.02, 0.04)):
            self.enabled = enabled
            self.reference = (0.0, 0.0, 0.0) if reference else None
            self.reference_rtk_flag = flag
            self.reference_accuracy = accuracy

    class _Chunk:
        def __init__(self, cameras):
            self.cameras = cameras

    def test_counts_fixed_and_float_solutions(self):
        chunk = self._Chunk([
            self._Camera(flag=50),
            self._Camera(flag=34),
            self._Camera(flag=0),
        ])
        stats = eo.camera_reference_stats(chunk)
        self.assertEqual(stats["cameras"], 3)
        self.assertEqual(stats["with_reference"], 3)
        self.assertEqual(stats["rtk_fixed"], 2)

    def test_a_disabled_camera_is_not_counted(self):
        chunk = self._Chunk([self._Camera(), self._Camera(enabled=False)])
        stats = eo.camera_reference_stats(chunk)
        self.assertEqual(stats["cameras"], 1)
        self.assertEqual(stats["with_reference"], 1)

    def test_a_camera_without_a_reference_is_not_counted(self):
        chunk = self._Chunk([self._Camera(), self._Camera(reference=False)])
        stats = eo.camera_reference_stats(chunk)
        self.assertEqual(stats["with_reference"], 1)

    def test_the_median_horizontal_accuracy_is_reported(self):
        # The horizontal sigma is hypot(x, y): 0.014, 0.042, 0.071 -> 0.042.
        chunk = self._Chunk([
            self._Camera(accuracy=(0.01, 0.01, 0.02)),
            self._Camera(accuracy=(0.03, 0.03, 0.06)),
            self._Camera(accuracy=(0.05, 0.05, 0.10)),
        ])
        stats = eo.camera_reference_stats(chunk)
        self.assertEqual(stats["median_horizontal_m"], "0.042")

    def test_no_accuracy_reports_a_dash(self):
        chunk = self._Chunk([self._Camera(accuracy=None)])
        stats = eo.camera_reference_stats(chunk)
        self.assertEqual(stats["median_horizontal_m"], "-")


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


class SpreadViewsTest(unittest.TestCase):
    """The refined set must span the station's observers, not cluster on one."""

    def views(self, cameras):
        return [{"camera": camera, "px": 0.0, "py": 0.0, "target_radius_px": 10.0}
                for camera in cameras]

    def test_the_seed_is_the_largest_apparent_target(self):
        # The seed is the most pixels on the target, so the most accurate centre.
        views = [
            {"camera": "a", "px": 0.0, "py": 0.0, "target_radius_px": 5.0},
            {"camera": "b", "px": 0.0, "py": 0.0, "target_radius_px": 20.0},
            {"camera": "c", "px": 0.0, "py": 0.0, "target_radius_px": 12.0},
        ]
        centres = {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0), "c": (2.0, 0.0, 0.0)}
        chosen = eo.spread_views(views, 1, centres)
        self.assertEqual([view["camera"] for view in chosen], ["b"])

    def test_the_next_pick_is_the_farthest_observer(self):
        # Seed b at x=1; the farthest remaining centre is c at x=100, not a at 0.
        views = self.views(["a", "b", "c"])
        centres = {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0), "c": (100.0, 0.0, 0.0)}
        chosen = eo.spread_views(views, 2, centres)
        self.assertEqual([view["camera"] for view in chosen], ["a", "c"])

    def test_a_cluster_is_broken_up(self):
        # Three near-coincident cameras and one far away: the far one must be
        # picked, which is what gives the triangulation a baseline.
        views = self.views(["a", "b", "c", "far"])
        centres = {"a": (0.0, 0.0, 0.0), "b": (0.1, 0.0, 0.0),
                   "c": (0.2, 0.0, 0.0), "far": (50.0, 0.0, 0.0)}
        chosen = eo.spread_views(views, 2, centres)
        self.assertIn("far", [view["camera"] for view in chosen])

    def test_fewer_views_than_the_limit_keeps_them_all(self):
        views = self.views(["a", "b"])
        centres = {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0)}
        self.assertEqual(len(eo.spread_views(views, 16, centres)), 2)

    def test_a_view_with_no_centre_is_never_preferred(self):
        # A view whose centre is unknown cannot contribute a baseline, so it is
        # only returned when there are not enough known-centre views.
        views = self.views(["a", "b", "unknown"])
        centres = {"a": (0.0, 0.0, 0.0), "b": (10.0, 0.0, 0.0)}
        chosen = eo.spread_views(views, 2, centres)
        self.assertNotIn("unknown", [view["camera"] for view in chosen])

    def test_select_refine_views_uses_the_centres_when_given(self):
        projections = {
            "GA1": self.views(["a", "b", "c"]),
        }
        centres = {"a": (0.0, 0.0, 0.0), "b": (0.1, 0.0, 0.0), "c": (40.0, 0.0, 0.0)}
        selected = eo.select_refine_views(projections, 2, centres)
        self.assertIn("c", [view["camera"] for view in selected["GA1"]])


class BaselineHeightRatioTest(unittest.TestCase):
    """The intersection-conditioning measure the exclusion rule tests."""

    def test_a_wide_baseline_at_short_range_is_well_conditioned(self):
        # Two cameras 10 m apart looking at a point 50 m above their midpoint.
        # The mean range is the 3-D distance, so the ratio is 10 / mean_range.
        centres = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        point = (5.0, 0.0, 50.0)
        ratio = eo.baseline_height_ratio(centres, point)
        mean_range = math.dist((0.0, 0.0, 0.0), point)
        self.assertAlmostEqual(ratio, 10.0 / mean_range, places=9)
        self.assertGreater(ratio, eo.MIN_BASELINE_HEIGHT_RATIO)

    def test_a_tight_cluster_at_long_range_is_poorly_conditioned(self):
        # Two consecutive frames 2 m apart at 55 m range: ratio ~0.036.
        centres = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        ratio = eo.baseline_height_ratio(centres, (1.0, 0.0, 55.0))
        self.assertLess(ratio, eo.MIN_BASELINE_HEIGHT_RATIO)

    def test_the_largest_pair_sets_the_baseline(self):
        # Three cameras; the widest pair (a-c, 20 m) sets the baseline, not a-b.
        centres = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        point = (10.0, 0.0, 100.0)
        ratio = eo.baseline_height_ratio(centres, point)
        mean_range = sum(math.dist(centre, point) for centre in centres) / 3.0
        self.assertAlmostEqual(ratio, 20.0 / mean_range, places=9)

    def test_fewer_than_two_centres_has_no_ratio(self):
        self.assertIsNone(eo.baseline_height_ratio([(0.0, 0.0, 0.0)], (0.0, 0.0, 1.0)))
        self.assertIsNone(eo.baseline_height_ratio([], (0.0, 0.0, 1.0)))


class IntersectionExclusionTest(unittest.TestCase):
    """The exclusion rule is fixed before results and reported with a reason."""

    def test_two_images_are_excluded(self):
        reason = eo.intersection_exclusion(
            ["a", "b"], [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], (5.0, 0.0, 50.0))
        self.assertIsNotNone(reason)
        self.assertIn("accepted image", reason)

    def test_three_well_spread_images_are_scored(self):
        reason = eo.intersection_exclusion(
            ["a", "b", "c"],
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 10.0, 0.0)],
            (5.0, 5.0, 50.0))
        self.assertIsNone(reason)

    def test_a_low_baseline_ratio_is_excluded(self):
        # Three images but all within 2 m at 55 m range: ratio ~0.036.
        reason = eo.intersection_exclusion(
            ["a", "b", "c"],
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
            (1.0, 0.0, 55.0))
        self.assertIsNotNone(reason)
        self.assertIn("baseline/height", reason)

    def test_the_thresholds_are_the_documented_values(self):
        self.assertEqual(eo.MIN_INTERSECTION_IMAGES, 3)
        self.assertAlmostEqual(eo.MIN_BASELINE_HEIGHT_RATIO, 0.1, places=9)


class RefineAndCollectTest(unittest.TestCase):
    """Every detected view is a candidate observation, not only the refined ones.

    The previous version accepted only the ``max_views`` refined views, so a
    station detected in 20 cameras but refined in 16 could never exceed 16
    observations. These fakes pin the corrected behaviour: a detected view that
    was not refined is still accepted, and a wrong-target detection is still
    rejected by the residual gate.
    """

    class _Projection:
        def __init__(self, camera, px, py):
            self.camera = type("Camera", (), {"label": camera})()
            self.pixel = (px, py)

    class _Marker:
        def __init__(self, label, position):
            self.label = label
            self.position = position
            self.projections = []

    class _Chunk:
        def __init__(self, triangulated):
            self.markers = []
            self.meta = {}
            self._triangulated = triangulated

        def removeMarkerProjection(self, label, camera):
            for marker in self.markers:
                if marker.label == label:
                    marker.projections = [projection for projection
                                          in marker.projections
                                          if projection.camera.label != camera]

        def setMarkerProjection(self, label, camera, px, py):
            for marker in self.markers:
                if marker.label == label:
                    for projection in marker.projections:
                        if projection.camera.label == camera:
                            projection.pixel = (px, py)
                            return
                    marker.projections.append(
                        RefineAndCollectTest._Projection(camera, px, py))

        def refineMarkers(self, markers, **_kwargs):
            # A moved position is what marks the marker as triangulated.
            for marker in self.markers:
                if marker.label in markers:
                    marker.position = self._triangulated

    def calibration(self):
        # A 200 px frame with f = 1: the point (0, 0, 10) projects to the image
        # centre (100, 100), so a detection there has a zero residual.
        return {"type": "frame", "width": 200, "height": 200, "f": 1.0,
                "cx": 0.0, "cy": 0.0, "b1": 0.0, "b2": 0.0,
                "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
                "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}

    def run_collect(self, detections, triangulated=(0.0, 0.0, 10.0)):
        chunk = self._Chunk(triangulated)
        marker = self._Marker("A1", (0.0, 0.0, 0.0))
        for camera, _px, _py in detections:
            marker.projections.append(self._Projection(camera, 100.0, 100.0))
        chunk.markers = [marker]
        stations = [{"label": "A1", "lat": 0.0, "lon": 0.0, "height": 0.0,
                     "role": "check_point"}]
        detected = {"A1": [{"camera": camera, "px": px, "py": py,
                            "target_radius_px": 10.0}
                           for camera, px, py in detections]}
        per_point = {"A1": [{"image": camera, "shift_px": 1.0,
                             "residual_px": None, "target_radius_px": 10.0,
                             "accepted": False, "detected_px": (px, py)}
                            for camera, px, py in detections]}
        calibration = self.calibration()
        calibrations = {camera: calibration for camera, _px, _py in detections}
        poses = {camera: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                 for camera, _px, _py in detections}
        centres = {camera: (float(index), 0.0, 0.0)
                   for index, (camera, _px, _py) in enumerate(detections)}
        return eo.refine_and_collect(
            chunk, stations, detected, per_point, {}, calibrations, poses,
            16, 60.0, centres=centres)

    def test_a_detected_view_beyond_the_refined_limit_is_still_accepted(self):
        # 20 detections, only 16 refined: all 20 are valid observations.
        detections = [(f"cam{index}", 100.0, 100.0) for index in range(20)]
        observations, _per_point, _meta = self.run_collect(detections)
        self.assertEqual(len(observations["A1"]), 20)

    def test_a_wrong_target_detection_is_rejected_by_the_residual(self):
        # cam19 detected a different target at (500, 500); the triangulated
        # position reprojects to (100, 100), so its residual is large.
        detections = [(f"cam{index}", 100.0, 100.0) for index in range(19)]
        detections.append(("cam19", 500.0, 500.0))
        observations, per_point, _meta = self.run_collect(detections)
        accepted = [row["image"] for row in observations["A1"]]
        self.assertNotIn("cam19", accepted)
        self.assertEqual(len(accepted), 19)
        cam19 = next(record for record in per_point["A1"]
                     if record["image"] == "cam19")
        self.assertGreater(cam19["residual_px"], eo.MIN_RESIDUAL_PX)

    def test_an_undetected_view_is_not_an_observation(self):
        # A projected view that DetectMarkers never found has no detected_px and
        # must not be written as an observation.
        detections = [(f"cam{index}", 100.0, 100.0) for index in range(3)]
        chunk = self._Chunk((0.0, 0.0, 10.0))
        marker = self._Marker("A1", (0.0, 0.0, 0.0))
        for camera, _px, _py in detections:
            marker.projections.append(self._Projection(camera, 100.0, 100.0))
        chunk.markers = [marker]
        stations = [{"label": "A1", "lat": 0.0, "lon": 0.0, "height": 0.0,
                     "role": "check_point"}]
        detected = {"A1": [{"camera": camera, "px": px, "py": py,
                            "target_radius_px": 10.0}
                           for camera, px, py in detections]}
        # One projected-but-undetected view: no detected_px.
        per_point = {"A1": [{"image": camera, "shift_px": 1.0,
                             "residual_px": None, "target_radius_px": 10.0,
                             "accepted": False, "detected_px": (px, py)}
                            for camera, px, py in detections]}
        per_point["A1"].append({"image": "cam_undetected", "shift_px": 0.0,
                                "residual_px": None, "target_radius_px": 10.0,
                                "accepted": False})
        calibration = self.calibration()
        calibrations = {camera: calibration for camera, _px, _py in detections}
        poses = {camera: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                 for camera, _px, _py in detections}
        centres = {camera: (float(index), 0.0, 0.0)
                   for index, (camera, _px, _py) in enumerate(detections)}
        observations, _per_point, _meta = eo.refine_and_collect(
            chunk, stations, detected, per_point, {}, calibrations, poses,
            16, 60.0, centres=centres)
        accepted = [row["image"] for row in observations["A1"]]
        self.assertNotIn("cam_undetected", accepted)
        self.assertEqual(len(accepted), 3)


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
        # A sheet with only one row has 47 grid cells missing, so the missing
        # cell cannot be identified uniquely and the row falls back to UNNAMED.
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                [None, 0.0, 0.0, 0.0, 55.262911463, -1.876660186, 222.2881,
                 0.0, 0.0, 171.0],
            ])
            stations = eo.read_stations(path)
        self.assertEqual(stations[0]["label"], eo.UNNAMED_STATION_LABEL)
        self.assertEqual(stations[0]["role"], "check_point")

    def test_the_one_missing_grid_cell_names_the_unnamed_row(self):
        # The published sheet carries every 6x8 grid cell but A1, and the
        # unlabelled row is that cell. The label is recovered from the labels the
        # sheet does carry, so the station is named A1 at the source.
        rows = []
        for label in eo.grid_labels():
            if label == "A1":
                rows.append([None, 0.0, 0.0, 0.0, 55.262911463, -1.876660186,
                             222.2881, 0.0, 0.0, 171.0])
            else:
                rows.append([label, 0.0, 0.0, 0.0, 55.26, -1.87, 220.0,
                             0.0, 0.0, 170.0])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, rows)
            stations = eo.read_stations(path)
        unnamed = [station for station in stations
                   if station["label"] == "A1"]
        self.assertEqual(len(unnamed), 1)
        self.assertEqual(unnamed[0]["role"], "check_point")
        self.assertNotIn(eo.UNNAMED_STATION_LABEL,
                         [station["label"] for station in stations])

    def test_infer_unnamed_label_returns_the_single_missing_cell(self):
        labels = [label for label in eo.grid_labels() if label != "A1"]
        self.assertEqual(eo.infer_unnamed_label(labels), "A1")

    def test_infer_unnamed_label_is_none_when_several_cells_are_missing(self):
        labels = [label for label in eo.grid_labels()
                  if label not in ("A1", "B2")]
        self.assertIsNone(eo.infer_unnamed_label(labels))

    def test_infer_unnamed_label_ignores_gcp_and_supplementary_labels(self):
        # GCP labels (GA1) and S1-S3 are not grid cells; they must not be
        # mistaken for the missing cell.
        labels = [label for label in eo.grid_labels() if label != "A1"]
        labels += ["GA1", "GA5", "S1", "S2", "S3"]
        self.assertEqual(eo.infer_unnamed_label(labels), "A1")

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
