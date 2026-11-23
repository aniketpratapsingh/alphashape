#!/usr/bin/env python3
"""Pure helpers behind the real-block dome and calibration diagnostics.

The real-block protocol (`engine/tests/run_gcp_validation.py`) fits a height
versus radius profile to the leave-one-out check points and to the 146 camera
residuals, and reports the calibration covariance. These tests pin the parts
that need no chunk: the least-squares fit (including the honest handling of an
underdetermined model), the convex-hull area used for the GCP-versus-block
coverage comparison, the fold-fixing of the quadratic surface, and the option
translation for the model-choice probes.

Plain unittest, no pytest.
"""

import importlib.util
import pathlib
import sys
import types
import unittest


def load_script():
    path = (pathlib.Path(__file__).resolve().parents[2] /
            "tests" / "run_gcp_validation.py")
    spec = importlib.util.spec_from_file_location("run_gcp_validation", path)
    module = importlib.util.module_from_spec(spec)
    # The script imports the built `alphashape` module at top level. These tests
    # exercise only pure helpers, so a stub satisfies the import when the module
    # is not built (the same trick the previous change used). When the real
    # module is importable it is used unchanged.
    if "alphashape" not in sys.modules:
        try:
            import alphashape  # noqa: F401, PLC0415
        except ImportError:
            sys.modules["alphashape"] = types.ModuleType("alphashape")
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()


class TestLeastSquares(unittest.TestCase):
    def test_recovers_a_radial_slope_and_reports_significance(self):
        # height = 0.01 + 0.002 * (r/100)^2, with a tiny deterministic wiggle so
        # the residual degrees of freedom are non-zero.
        rows = []
        values = []
        for radius, wiggle in zip((0.0, 100.0, 200.0, 300.0, 400.0),
                                  (0.0001, -0.0001, 0.0001, -0.0001, 0.0)):
            scaled = radius / SCRIPT.DOME_RADIUS_SCALE_M
            rows.append([scaled * scaled])
            values.append(0.01 + 0.002 * scaled * scaled + wiggle)
        fit = SCRIPT.linear_least_squares(
            rows, values, ["radius_100m_squared"])
        self.assertEqual(fit["status"], "ok")
        self.assertEqual(fit["samples"], 5)
        self.assertEqual(fit["degrees_of_freedom"], 3)
        slope = fit["by_name"]["radius_100m_squared"]
        # The tiny wiggle shifts the estimate by a few parts per million; the
        # noise floor is what keeps the standard error from being exactly zero.
        self.assertAlmostEqual(slope["coefficient"], 0.002, delta=1e-5)
        self.assertIsNotNone(slope["standard_error"])
        self.assertGreater(abs(slope["t"]), 10.0)
        self.assertGreater(fit["r_squared"], 0.999)

    def test_underdetermined_is_reported_not_extrapolated(self):
        # The full quadratic surface has six terms; five check points cannot
        # identify it, so no coefficients are invented.
        rows = [[1.0, 1.0, 1.0, 1.0, 1.0]] * 5
        fit = SCRIPT.linear_least_squares(
            rows, [0.1] * 5,
            ["east_100m", "north_100m", "east_100m_squared",
             "east_100m_north_100m", "north_100m_squared"])
        self.assertEqual(fit["status"], "underdetermined")
        self.assertNotIn("coefficients", fit)

    def test_singular_predictors_are_reported(self):
        # A duplicated column is exactly singular.
        duplicated = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        fit = SCRIPT.linear_least_squares(duplicated, [1.0, 2.0, 3.0, 4.0],
                                          ["x", "x_again"])
        self.assertEqual(fit["status"], "singular")


class TestConvexHullArea(unittest.TestCase):
    def test_square_triangle_and_degenerate_inputs(self):
        self.assertAlmostEqual(
            SCRIPT.convex_hull_area([(0, 0), (2, 0), (2, 2), (0, 2)]), 4.0)
        self.assertAlmostEqual(
            SCRIPT.convex_hull_area([(0, 0), (2, 0), (0, 2), (0.5, 0.5)]), 2.0)
        self.assertEqual(SCRIPT.convex_hull_area([(0, 0), (1, 1), (2, 2)]), 0.0)
        self.assertEqual(SCRIPT.convex_hull_area([(0, 0), (1, 1)]), 0.0)


class TestDomeSummary(unittest.TestCase):
    def _fold(self, label, radius, vertical):
        point = {
            "label": label,
            "horizontal_m": 0.05,
            "vertical_m": vertical,
            "vertical_enu_m": vertical,
            "east_from_centroid_m": radius,
            "north_from_centroid_m": 0.0,
            "radius_m": radius,
            "radius_squared_m2": radius * radius,
        }
        return {"check": label, "dome": {
            "check_point": point,
            "camera": None,
            "gcp_hull_area_m2": 1000.0,
            "block_hull_area_m2": 100000.0,
        }}

    def test_five_check_points_fit_the_radial_term_only(self):
        folds = [self._fold(str(index), radius,
                            0.01 + 0.002 * (radius / 100.0) ** 2 +
                            (0.0001 if index % 2 else -0.0001))
                 for index, radius in enumerate((0.0, 100.0, 200.0, 340.0, 410.0))]
        summary = SCRIPT.summarise_dome(folds)
        radial = summary["check_fit_radial"]
        self.assertEqual(radial["status"], "ok")
        self.assertAlmostEqual(
            radial["by_name"]["radius_100m_squared"]["coefficient"],
            0.002, places=5)
        # The six-term surface is still underdetermined at five points.
        self.assertEqual(summary["check_fit_quadratic"]["status"],
                         "underdetermined")
        self.assertAlmostEqual(summary["gcp_block_area_ratio"], 0.01)

    def test_no_dome_records_returns_none(self):
        self.assertIsNone(SCRIPT.summarise_dome([{"check": "A"}]))


class TestVariantOptions(unittest.TestCase):
    def test_model_choice_flags_reach_optimize_cameras(self):
        options = SCRIPT.optimize_options(
            {"fit": {"f": False, "k1": True, "p1": False}})
        self.assertTrue(options["use_gcps"])
        self.assertFalse(options["fit_f"])
        self.assertTrue(options["fit_k1"])
        self.assertFalse(options["fit_p1"])

    def test_every_variant_names_its_lens_model(self):
        # A variant that does not ask for the adaptive model pins the plain
        # one, so its row keeps its meaning now that OptimizeCameras selects
        # the adaptive model by itself on a controlled block.
        plain = SCRIPT.optimize_options({})
        self.assertIs(plain["adaptive_fitting_auto"], False)
        self.assertNotIn("adaptive_fitting", plain)
        adaptive = SCRIPT.optimize_options({"adaptive_fitting": True})
        self.assertTrue(adaptive["adaptive_fitting"])
        self.assertNotIn("adaptive_fitting_auto", adaptive)
        # Only the engine_defaults variant leaves the engine's own default.
        defaults = SCRIPT.optimize_options({"engine_defaults": True})
        self.assertEqual(defaults, {"use_gcps": True})

    def test_engine_defaults_wins_over_an_explicit_adaptive_request(self):
        # The gated variant must run exactly the shipped default: no fit_*
        # permission set and no adaptive_fitting_auto override, even if a
        # caller also names adaptive_fitting.
        options = SCRIPT.optimize_options(
            {"engine_defaults": True, "adaptive_fitting": True})
        self.assertEqual(options, {"use_gcps": True})

    def test_doming_variants_are_registered(self):
        names = {name for name, _, _ in SCRIPT.DIAGNOSTIC_VARIANTS}
        for name in ("camera_rtk_k1234", "camera_rtk_f_fixed",
                     "camera_rtk_p_fixed", "fixed_calibration_camera_rtk",
                     "camera_rtk_per_image_weight_x2",
                     "camera_rtk_per_image_weight_x5"):
            self.assertIn(name, names)

    def test_fixed_calibration_variant_requests_the_controlled_source(self):
        options = {name: option for name, _, option
                   in SCRIPT.DIAGNOSTIC_VARIANTS}["fixed_calibration_camera_rtk"]
        self.assertTrue(options["fixed_calibration_from_control"])
        self.assertTrue(options["calibration_source_adaptive"])
        # The camera-RTK pass itself must not re-run adaptive selection with a
        # fully fixed calibration.
        self.assertNotIn("adaptive_fitting", options)


class TestMatchParams(unittest.TestCase):
    def test_values_are_typed_for_the_task(self):
        self.assertEqual(
            SCRIPT.parse_match_params(["tiepoint_limit=8000",
                                       "generic_preselection=false",
                                       "reference_preselection_mode=sequential"]),
            {"tiepoint_limit": 8000, "generic_preselection": False,
             "reference_preselection_mode": "sequential"})

    def test_bad_pairs_are_refused(self):
        for pairs in (["tiepoint_limit"], ["=1"], ["tiepoint_limit="],
                      ["a=1", "a=2"], ["keypoint_limit_per_mpx=1000"]):
            with self.assertRaises(ValueError, msg=pairs):
                SCRIPT.parse_match_params(pairs)

    def test_the_flag_reaches_the_arguments(self):
        args = SCRIPT.parse_args(["--real", "/x", "--match-param", "tiepoint_limit=8000"])
        self.assertEqual(args.match_params, {"tiepoint_limit": 8000})
        self.assertEqual(SCRIPT.parse_args(["--real", "/x"]).match_params, {})


class TestControlCheckSplit(unittest.TestCase):
    """The report-only control/check split: control is --gcp-list, checks are
    the --check-list stations not already control."""

    def _station(self, label):
        return {"label": label, "lon": 0.0, "lat": 0.0, "height": 0.0,
                "observations": []}

    def test_checks_are_the_stations_not_in_control(self):
        control = [self._station("A"), self._station("B")]
        stations = [self._station("A"), self._station("B"),
                    self._station("C"), self._station("D")]
        controls, checks = SCRIPT.split_control_check(control, stations)
        self.assertEqual([gcp["label"] for gcp in controls], ["A", "B"])
        self.assertEqual([gcp["label"] for gcp in checks], ["C", "D"])

    def test_a_station_in_both_lists_is_control_not_a_check(self):
        # A point that constrains the fit must never also be scored as a check.
        control = [self._station("A")]
        stations = [self._station("A"), self._station("B")]
        _, checks = SCRIPT.split_control_check(control, stations)
        self.assertEqual([gcp["label"] for gcp in checks], ["B"])

    def test_control_is_every_gcp_list_station(self):
        control = [self._station("A"), self._station("B")]
        _, checks = SCRIPT.split_control_check(control, [self._station("A")])
        self.assertEqual(checks, [])

    def test_the_check_list_argument_is_optional(self):
        self.assertIsNone(SCRIPT.parse_args(["--real", "/x"]).check_list)
        args = SCRIPT.parse_args(["--real", "/x", "--check-list", "/y"])
        self.assertEqual(args.check_list, "/y")


class _Calibration:
    """A frame calibration with the attributes the projector reads."""

    def __init__(self, **overrides):
        self.type = "frame"
        self.width = 200
        self.height = 200
        self.f = 1.0
        self.cx = 0.0
        self.cy = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0
        for key, value in overrides.items():
            setattr(self, key, value)


IDENTITY_POSE = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]


class TestProjectMarkerPixel(unittest.TestCase):
    """The forward Brown model the marker reprojection RMS is measured with."""

    def test_a_point_on_the_axis_lands_at_the_image_centre(self):
        pixel = SCRIPT.project_marker_pixel((0.0, 0.0, 10.0), IDENTITY_POSE,
                                            _Calibration())
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], 100.0, places=9)
        self.assertAlmostEqual(pixel[1], 100.0, places=9)

    def test_a_point_behind_the_camera_has_no_projection(self):
        self.assertIsNone(SCRIPT.project_marker_pixel(
            (0.0, 0.0, -10.0), IDENTITY_POSE, _Calibration()))

    def test_the_principal_point_offset_shifts_the_projection(self):
        pixel = SCRIPT.project_marker_pixel(
            (0.0, 0.0, 10.0), IDENTITY_POSE, _Calibration(cx=10.0, cy=-5.0))
        self.assertAlmostEqual(pixel[0], 110.0, places=9)
        self.assertAlmostEqual(pixel[1], 95.0, places=9)

    def test_a_fisheye_sensor_is_refused(self):
        with self.assertRaises(ValueError):
            SCRIPT.project_marker_pixel((0.0, 0.0, 10.0), IDENTITY_POSE,
                                        _Calibration(type="fisheye"))


class _Projection:
    def __init__(self, camera, px, py):
        self.camera = type("Camera", (), {"label": camera})()
        self.pixel = (px, py)


class _Camera:
    def __init__(self, label, sensor_id=0):
        self.label = label
        self.sensor_id = sensor_id
        self.pose = IDENTITY_POSE


class _Sensor:
    def __init__(self, calibration):
        self.calibration = calibration


class _Marker:
    def __init__(self, label, position, projections):
        self.label = label
        self.position = position
        self.projections = projections


class _Chunk:
    def __init__(self, cameras, sensors, markers):
        self.cameras = cameras
        self.sensors = sensors
        self.markers = markers


class TestMarkerReprojectionRms(unittest.TestCase):
    """The post-adjustment marker reprojection RMS, per point and overall."""

    def test_a_perfect_reprojection_has_zero_rms(self):
        # The marker at (0, 0, 10) reprojects to (100, 100); the measured pixel
        # is exactly there, so the residual is zero.
        chunk = _Chunk(
            cameras=[_Camera("a"), _Camera("b")],
            sensors=[_Sensor(_Calibration())],
            markers=[_Marker("A1", (0.0, 0.0, 10.0),
                             [_Projection("a", 100.0, 100.0),
                              _Projection("b", 100.0, 100.0)])])
        per_label, overall = SCRIPT.marker_reprojection_rms(chunk, ["A1"])
        self.assertEqual(per_label["A1"]["observations"], 2)
        self.assertAlmostEqual(per_label["A1"]["rms_px"], 0.0, places=9)
        self.assertAlmostEqual(overall, 0.0, places=9)

    def test_a_known_offset_gives_the_expected_rms(self):
        # Both measured pixels are 3 px off in x, so the RMS is 3 px.
        chunk = _Chunk(
            cameras=[_Camera("a"), _Camera("b")],
            sensors=[_Sensor(_Calibration())],
            markers=[_Marker("A1", (0.0, 0.0, 10.0),
                             [_Projection("a", 103.0, 100.0),
                              _Projection("b", 103.0, 100.0)])])
        per_label, overall = SCRIPT.marker_reprojection_rms(chunk, ["A1"])
        self.assertAlmostEqual(per_label["A1"]["rms_px"], 3.0, places=9)
        self.assertAlmostEqual(overall, 3.0, places=9)

    def test_a_projection_onto_a_missing_camera_is_skipped(self):
        # A projection whose camera is not in the chunk is not counted.
        chunk = _Chunk(
            cameras=[_Camera("a")],
            sensors=[_Sensor(_Calibration())],
            markers=[_Marker("A1", (0.0, 0.0, 10.0),
                             [_Projection("a", 100.0, 100.0),
                              _Projection("gone", 500.0, 500.0)])])
        per_label, _overall = SCRIPT.marker_reprojection_rms(chunk, ["A1"])
        self.assertEqual(per_label["A1"]["observations"], 1)
        self.assertAlmostEqual(per_label["A1"]["rms_px"], 0.0, places=9)

    def test_a_marker_with_no_position_has_no_rms(self):
        chunk = _Chunk(
            cameras=[_Camera("a")],
            sensors=[_Sensor(_Calibration())],
            markers=[_Marker("A1", None, [_Projection("a", 100.0, 100.0)])])
        per_label, overall = SCRIPT.marker_reprojection_rms(chunk, ["A1"])
        self.assertEqual(per_label["A1"]["observations"], 0)
        self.assertIsNone(per_label["A1"]["rms_px"])
        self.assertIsNone(overall)

    def test_only_the_requested_labels_are_measured(self):
        chunk = _Chunk(
            cameras=[_Camera("a")],
            sensors=[_Sensor(_Calibration())],
            markers=[_Marker("A1", (0.0, 0.0, 10.0),
                             [_Projection("a", 100.0, 100.0)]),
                     _Marker("B2", (0.0, 0.0, 10.0),
                             [_Projection("a", 100.0, 100.0)])])
        per_label, _overall = SCRIPT.marker_reprojection_rms(chunk, ["A1"])
        self.assertEqual(set(per_label), {"A1"})


if __name__ == "__main__":
    unittest.main()
