"""Tests for the OpenDroneMap leave-one-out reference helpers (no ODM needed)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import odm_reference  # noqa: E402

GCP_LIST = """EPSG:4326
16.47592207\t47.64350399\t463.662\t2501.0\t2452.0\tA.JPG\t1
16.47592207\t47.64350399\t463.662\t3185.9\t2514.8\tB.JPG\t1
16.47619479\t47.64346579\t461.732\t2000.0\t1000.0\tA.JPG\t2
16.47599194\t47.64329734\t461.463\t1500.0\t1700.0\tC.JPG\t3
16.47599194\t47.64329734\t461.463\t1600.0\t1800.0\tD.JPG\t3
"""


def look_at(centre, target):
    """OpenSfM world->camera rotation (axis-angle) and translation for a camera at
    ``centre`` looking at ``target`` with image y pointing roughly south."""
    centre = np.asarray(centre, float)
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.vstack([x, y, z])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    if angle < 1e-12:
        axis_angle = np.zeros(3)
    else:
        axis = np.array([rotation[2, 1] - rotation[1, 2],
                         rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(angle))
        axis_angle = axis * angle
    return axis_angle.tolist(), (-rotation @ centre).tolist()


class FoldTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.gcp = self.root / "gcp_list.txt"
        self.gcp.write_text(GCP_LIST, encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_points_are_grouped_by_coordinates_and_sorted_by_label(self):
        header, points = odm_reference.read_gcp_list(self.gcp)
        self.assertEqual(header, "EPSG:4326")
        self.assertEqual([p["label"] for p in points], ["1", "2", "3"])
        self.assertEqual(len(points[0]["observations"]), 2)

    def test_a_fold_leaves_exactly_its_point_out(self):
        holdout = odm_reference.write_fold(self.gcp, 3, self.root / "f.txt", self.root / "h.json")
        written = (self.root / "f.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(written[0], "EPSG:4326")
        self.assertEqual(len(written), 4)
        self.assertTrue(all(line.split()[6] in ("1", "2") for line in written[1:]))
        self.assertEqual(holdout["check"]["label"], "3")
        self.assertEqual(len(holdout["check"]["observations"]), 2)
        self.assertEqual(json.loads((self.root / "h.json").read_text())["fold"], 3)

    def test_a_fold_outside_the_points_is_refused(self):
        with self.assertRaises(SystemExit):
            odm_reference.write_fold(self.gcp, 4, self.root / "f.txt", self.root / "h.json")


class GeometryTest(unittest.TestCase):
    def test_the_reference_is_the_origin(self):
        enu = odm_reference.topocentric_from_lla(47.6, 16.4, 0.0, 47.6, 16.4, 0.0)
        for value in enu:
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_north_and_up_have_the_right_scale(self):
        # One arc-second of latitude is ~30.9 m at 47.6 N.
        enu = odm_reference.topocentric_from_lla(47.6 + 1 / 3600, 16.4, 10.0, 47.6, 16.4, 0.0)
        self.assertAlmostEqual(enu[0], 0.0, places=4)
        self.assertAlmostEqual(enu[1], 30.88, delta=0.05)
        self.assertAlmostEqual(enu[2], 10.0, delta=0.001)

    def test_normalization_matches_opensfm(self):
        u, v = odm_reference.normalized_from_pixel(-0.5, -0.5, 4000, 3000)
        self.assertAlmostEqual(u, -0.5)
        self.assertAlmostEqual(v, -0.375)

    def test_bearing_inverts_projection_for_every_model(self):
        cameras = [
            {"projection_type": "perspective", "width": 4000, "height": 3000,
             "focal": 0.85, "k1": -0.05, "k2": 0.01},
            {"projection_type": "brown", "width": 4000, "height": 3000,
             "focal_x": 0.85, "focal_y": 0.851, "c_x": 0.004, "c_y": -0.003,
             "k1": -0.05, "k2": 0.01, "k3": -0.002, "p1": 0.0007, "p2": -0.0004},
        ]
        for data in cameras:
            camera = odm_reference.Camera(data)
            point = np.array([0.31, -0.22, 1.0])
            u, v = camera.project(point)
            bearing = camera.bearing(u, v)
            np.testing.assert_allclose(bearing, point, atol=1e-9, err_msg=data["projection_type"])

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(SystemExit):
            odm_reference.Camera({"projection_type": "spherical", "width": 1, "height": 1})


class TriangulationTest(unittest.TestCase):
    def reconstruction(self, target):
        camera = {"projection_type": "brown", "width": 5472, "height": 3648,
                  "focal_x": 0.78, "focal_y": 0.78, "c_x": 0.002, "c_y": -0.001,
                  "k1": -0.01, "k2": 0.005, "k3": 0.0, "p1": 0.0002, "p2": 0.0001}
        shots = {}
        observations = []
        model = odm_reference.Camera(camera)
        size = max(camera["width"], camera["height"])
        for index, centre in enumerate(([-30.0, 0.0, 80.0], [30.0, 5.0, 82.0], [0.0, -35.0, 79.0])):
            rotation, translation = look_at(centre, [0.0, 0.0, 0.0])
            name = f"I{index}.JPG"
            shots[name] = {"camera": "c", "rotation": rotation, "translation": translation}
            r = odm_reference.rotation_from_axis_angle(rotation)
            u, v = model.project(r @ np.asarray(target) + np.asarray(translation))
            observations.append({"image": name,
                                 "px": u * size + camera["width"] / 2.0 - 0.5,
                                 "py": v * size + camera["height"] / 2.0 - 0.5})
        observations.append({"image": "NOT_RECONSTRUCTED.JPG", "px": 1.0, "py": 1.0})
        return {"cameras": {"c": camera}, "shots": shots}, observations

    def test_an_exact_point_is_recovered(self):
        target = [1.5, -2.0, 0.3]
        reconstruction, observations = self.reconstruction(target)
        result = odm_reference.triangulate(observations, reconstruction)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["views"], 3)
        np.testing.assert_allclose(result["point"], target, atol=1e-6)
        self.assertLess(result["reprojection_rms_px"], 1e-4)

    def test_one_view_is_not_enough(self):
        reconstruction, observations = self.reconstruction([0.0, 0.0, 0.0])
        result = odm_reference.triangulate(observations[:1], reconstruction)
        self.assertEqual(result["status"], "too_few_views")


class SummaryTest(unittest.TestCase):
    def result(self, mode, fold, h, v, status="ok", control_v=0.01, agreement=None):
        check = {"label": str(fold), "status": status}
        if status == "ok":
            check.update({"horizontal_m": h, "vertical_m": v, "views": 7,
                          "reprojection_rms_px": 0.5})
        result = {"mode": mode, "fold": fold, "check": check,
                  "control": [{"status": "ok", "horizontal_m": 0.01,
                               "vertical_m": control_v}],
                  "shots_reconstructed": 176}
        if agreement is not None:
            result["odm_agreement"] = agreement
        return result

    def test_rms_per_mode_and_the_target(self):
        results = [self.result("gcp_only", 1, 0.03, 0.04), self.result("gcp_only", 2, 0.04, -0.03),
                   self.result("gcp_gps", 1, 0.08, 0.1)]
        summary, markdown = odm_reference.summarize(results, 0.05)
        only = summary["modes"]["gcp_only"]
        self.assertAlmostEqual(only["horizontal_rms_m"], math.sqrt((0.03 ** 2 + 0.04 ** 2) / 2))
        self.assertTrue(only["meets_target"])
        self.assertFalse(summary["modes"]["gcp_gps"]["meets_target"])
        self.assertIn("## gcp_only", markdown)

    def test_a_missing_fold_never_meets_the_target(self):
        results = [self.result("gcp_only", 1, 0.01, 0.01),
                   self.result("gcp_only", 2, 0, 0, status="too_few_views")]
        summary, _ = odm_reference.summarize(results, 0.05)
        self.assertFalse(summary["modes"]["gcp_only"]["meets_target"])
        self.assertEqual(len(summary["modes"]["gcp_only"]["missing"]), 1)

    def test_control_vertical_rms_is_reported(self):
        # The control vertical RMS is the diagnostic that tells a triangulation
        # bug from a block that simply does not fit its survey.
        results = [self.result("gcp_only", 1, 0.09, 0.4, control_v=0.12),
                   self.result("gcp_only", 2, 0.09, -0.4, control_v=0.10)]
        summary, markdown = odm_reference.summarize(results, 0.05)
        entry = summary["modes"]["gcp_only"]
        self.assertAlmostEqual(entry["control_vertical_rms_m"],
                               math.sqrt((0.12 ** 2 + 0.10 ** 2) / 2))
        self.assertIn("Control v RMS", markdown)
        self.assertIn("Control points, measured the same way", markdown)

    def test_odm_agreement_is_aggregated(self):
        agreement = {"points": 4, "horizontal_rms_m": 0.001, "vertical_rms_m": 0.02,
                     "max_horizontal_m": 0.002, "max_vertical_m": 0.03}
        results = [self.result("gcp_only", 1, 0.09, 0.4, agreement=agreement),
                   self.result("gcp_only", 2, 0.09, -0.4, agreement=agreement)]
        summary, markdown = odm_reference.summarize(results, 0.05)
        entry = summary["modes"]["gcp_only"]["odm_agreement"]
        self.assertEqual(entry["folds"], 2)
        self.assertAlmostEqual(entry["vertical_rms_m"], 0.02)
        self.assertIn("Agreement with ODM's own control points", markdown)

    def test_a_mode_without_agreement_omits_the_line(self):
        results = [self.result("gcp_strong", 1, 0.09, 0.4)]
        summary, markdown = odm_reference.summarize(results, 0.05)
        self.assertIsNone(summary["modes"]["gcp_strong"]["odm_agreement"])
        self.assertNotIn("Agreement with ODM", markdown)

    def test_pinned_control_points_are_not_read_as_a_perfect_triangulation(self):
        # gcp_strong pins the control points, so their residual is zero by
        # construction; the summary must say so rather than imply the
        # triangulation is exact.
        results = [self.result("gcp_strong", 1, 0.09, 0.4, control_v=0.0)]
        results[0]["control"][0]["horizontal_m"] = 0.0
        _, markdown = odm_reference.summarize(results, 0.05)
        self.assertIn("pinned by this mode's weights", markdown)
        self.assertNotIn("Control points, measured the same way: 0.00 cm horizontal", markdown)


class OdmAgreementTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.opensfm = Path(self.directory.name)
        (self.opensfm / "stats").mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def write_stats(self, points):
        (self.opensfm / "stats" / "ground_control_points.json").write_text(
            json.dumps(points), encoding="utf-8")

    def test_our_points_are_compared_with_odms(self):
        self.write_stats([
            {"id": "1", "coordinates": [1.0, 2.0, 3.0]},
            {"id": "2", "coordinates": [4.0, 5.0, 6.0]},
        ])
        control = [
            {"label": "1", "status": "ok", "point": [1.01, 2.0, 3.02]},
            {"label": "2", "status": "ok", "point": [4.0, 5.0, 5.98]},
        ]
        agreement = odm_reference.odm_agreement(
            self.opensfm, control, "reconstruction.topocentric.json")
        self.assertEqual(agreement["points"], 2)
        self.assertAlmostEqual(agreement["max_horizontal_m"], 0.01)
        self.assertAlmostEqual(agreement["max_vertical_m"], 0.02)

    def test_a_missing_stats_file_is_not_an_error(self):
        control = [{"label": "1", "status": "ok", "point": [0.0, 0.0, 0.0]}]
        self.assertIsNone(odm_reference.odm_agreement(
            self.opensfm, control, "reconstruction.topocentric.json"))

    def test_a_rerun_reconstruction_is_not_compared(self):
        # ODM's statistics describe the topocentric block; a re-run
        # reconstruction leaves them stale, so the comparison is skipped.
        self.write_stats([{"id": "1", "coordinates": [0.0, 0.0, 0.0]}])
        control = [{"label": "1", "status": "ok", "point": [0.0, 0.0, 0.0]}]
        self.assertIsNone(odm_reference.odm_agreement(
            self.opensfm, control, "reconstruction.json"))

    def test_a_point_odm_did_not_measure_is_skipped(self):
        self.write_stats([{"id": "1", "coordinates": [0.0, 0.0, 0.0]}])
        control = [{"label": "1", "status": "ok", "point": [0.0, 0.0, 0.0]},
                   {"label": "9", "status": "ok", "point": [1.0, 1.0, 1.0]}]
        agreement = odm_reference.odm_agreement(
            self.opensfm, control, "reconstruction.topocentric.json")
        self.assertEqual(agreement["points"], 1)


if __name__ == "__main__":
    unittest.main()
