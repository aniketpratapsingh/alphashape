#!/usr/bin/env python3
"""Reference-frame helpers behind the real-block GCP protocol.

The protocol (`engine/tests/run_gcp_validation.py`) tests whether the camera
RTK positions and the surveyed GCPs are in the same realization of the geodetic
system. These tests pin the parts that need no chunk: the 3-parameter
translation and 7-parameter Helmert fit between the adjusted camera centres and
the stored references, the direction of the empirical correction (a measured
``reference - centre`` offset is removed), the GPS-week to decimal-year
conversion used for the coordinate epoch, and -- when pyproj is installed --
the published ETRS89 <-> ITRF2014 offset at this dataset's site and epoch.

Plain unittest, no pytest.
"""

import importlib.util
import math
import pathlib
import unittest

PYPROJ = importlib.util.find_spec("pyproj") is not None


def load_script():
    path = (pathlib.Path(__file__).resolve().parents[2] /
            "tests" / "run_gcp_validation.py")
    spec = importlib.util.spec_from_file_location("run_gcp_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()

# The dataset's site (Helenenschacht) and capture epoch from its RTK event
# marks (GPS week 2211, time of week 296288.242869 s).
SITE_LON = 16.47592207
SITE_LAT = 47.64350399
SITE_HEIGHT = 463.662
CAPTURE_EPOCH = 2022.3956960376354


class _Camera:
    def __init__(self, label, centre_ecef, reference):
        self.label = label
        self.center = centre_ecef
        self.reference = reference
        self.enabled = True


class _Chunk:
    """Only the attributes the frame fit reads; identity local-to-ECEF."""

    def __init__(self, cameras):
        self.cameras = cameras
        self.transform = [[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]]


def _enu_to_ecef(enu, origin):
    lon, lat, height = SCRIPT.enu_to_geodetic(enu, origin)
    return SCRIPT.geodetic_to_ecef(lon, lat, height)


def _synthetic_block(translation, rotation_rad, scale_mu):
    """Cameras scattered around the site, with a known Helmert applied.

    The adjusted centre is the truth ``c``; the stored reference is
    ``a = c + t + theta x c + mu c`` in ENU, so the fit has to recover those
    parameters. Points span a 80 m square at three heights, like a nadir block.
    """
    origin = (SITE_LON, SITE_LAT, SITE_HEIGHT)
    points = []
    for east in (-40.0, -20.0, 0.0, 20.0, 40.0):
        for north in (-40.0, -20.0, 0.0, 20.0, 40.0):
            for up in (-2.0, 0.0, 2.0):
                points.append((east, north, up))
    cameras = []
    for index, centre in enumerate(points):
        offset = [
            translation[0] + scale_mu * centre[0]
            + rotation_rad[1] * centre[2] - rotation_rad[2] * centre[1],
            translation[1] + scale_mu * centre[1]
            + rotation_rad[2] * centre[0] - rotation_rad[0] * centre[2],
            translation[2] + scale_mu * centre[2]
            + rotation_rad[0] * centre[1] - rotation_rad[1] * centre[0],
        ]
        centre_ecef = _enu_to_ecef(centre, origin)
        reference_ecef = _enu_to_ecef(
            [centre[axis] + offset[axis] for axis in range(3)], origin)
        lat, lon, height = SCRIPT.ecef_to_geodetic(*reference_ecef)
        cameras.append(_Camera("c{:03d}".format(index), list(centre_ecef),
                               (lon, lat, height)))
    return _Chunk(cameras), origin


class TestCameraReferenceFrameFit(unittest.TestCase):
    def test_translation_only_offset_is_recovered(self):
        chunk, origin = _synthetic_block((0.70, -0.49, 0.04), (0.0, 0.0, 0.0),
                                         0.0)
        fit = SCRIPT.camera_reference_frame_fit(chunk, origin)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["translation_m"][0], 0.70, delta=1e-3)
        self.assertAlmostEqual(fit["translation_m"][1], -0.49, delta=1e-3)
        self.assertAlmostEqual(fit["translation_m"][2], 0.04, delta=1e-3)
        self.assertLess(fit["translation_residual_rms_m"], 1e-3)
        self.assertLess(fit["helmert_residual_rms_m"], 1e-3)

    def test_helmert_recovers_rotation_and_scale(self):
        # 10 arcsec about up, 4 arcsec about east, -2 about north; 5 ppm scale.
        rotation = [4.0 / SCRIPT.ARCSEC_PER_RAD,
                    -2.0 / SCRIPT.ARCSEC_PER_RAD,
                    10.0 / SCRIPT.ARCSEC_PER_RAD]
        chunk, origin = _synthetic_block((0.70, -0.49, 0.04), rotation, 5e-6)
        fit = SCRIPT.camera_reference_frame_fit(chunk, origin)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["rotation_arcsec"][0], 4.0, delta=1e-3)
        self.assertAlmostEqual(fit["rotation_arcsec"][1], -2.0, delta=1e-3)
        self.assertAlmostEqual(fit["rotation_arcsec"][2], 10.0, delta=1e-3)
        self.assertAlmostEqual(fit["scale_ppm"], 5.0, delta=1e-2)
        self.assertLess(fit["helmert_residual_rms_m"], 1e-6)
        # A rotation/scale the translation alone cannot absorb.
        self.assertGreater(fit["translation_residual_rms_m"], 1e-4)

    def test_too_few_cameras_is_reported_not_fitted(self):
        chunk = _Chunk([])
        fit = SCRIPT.camera_reference_frame_fit(
            chunk, (SITE_LON, SITE_LAT, SITE_HEIGHT))
        self.assertEqual(fit["status"], "underdetermined")
        self.assertNotIn("translation_m", fit)


class TestFrameShiftSign(unittest.TestCase):
    def test_measured_offset_is_removed_from_the_reference(self):
        origin = (SITE_LON, SITE_LAT, SITE_HEIGHT)
        reference = SCRIPT.enu_to_geodetic((10.0, -5.0, 2.0), origin)
        shift = (10.0, -5.0, 2.0)  # reference minus the datum position
        lon, lat, height = SCRIPT.remove_frame_shift(*reference, shift)
        self.assertAlmostEqual(lon, SITE_LON, delta=1e-9)
        self.assertAlmostEqual(lat, SITE_LAT, delta=1e-9)
        self.assertAlmostEqual(height, SITE_HEIGHT, delta=1e-3)

    def test_zero_shift_is_the_identity(self):
        origin = (SITE_LON, SITE_LAT, SITE_HEIGHT)
        reference = SCRIPT.enu_to_geodetic((3.0, 4.0, -1.0), origin)
        shifted = SCRIPT.remove_frame_shift(*reference, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(shifted[0], reference[0], delta=1e-9)
        self.assertAlmostEqual(shifted[1], reference[1], delta=1e-9)
        self.assertAlmostEqual(shifted[2], reference[2], delta=1e-6)


class TestCaptureEpoch(unittest.TestCase):
    def test_gps_week_and_time_of_week(self):
        # The first usable event mark of the dataset's .mrk file.
        epoch = SCRIPT.decimal_year_from_gps(2211, 296288.242869)
        self.assertAlmostEqual(epoch, CAPTURE_EPOCH, delta=1e-6)
        # 2022-05-25 10:17:50 UTC
        self.assertGreater(epoch, 2022.39)
        self.assertLess(epoch, 2022.41)


class TestGcpFrameOrigin(unittest.TestCase):
    def test_check_point_is_excluded(self):
        gcps = [{"lon": 0.0, "lat": 0.0, "height": 0.0},
                {"lon": 2.0, "lat": 2.0, "height": 2.0},
                {"lon": 100.0, "lat": 100.0, "height": 100.0}]
        origin = SCRIPT.gcp_frame_origin(gcps, check_index=2)
        self.assertEqual(origin, (1.0, 1.0, 1.0))
        self.assertEqual(SCRIPT.gcp_frame_origin(gcps), (34.0, 34.0, 34.0))


@unittest.skipUnless(PYPROJ, "pyproj is not installed")
class TestPublishedFrameOffset(unittest.TestCase):
    def test_itrf2014_to_etrs89_at_the_dataset_site(self):
        gcps = [{"lon": SITE_LON, "lat": SITE_LAT, "height": SITE_HEIGHT}]
        offsets = SCRIPT.published_frame_offsets(gcps, CAPTURE_EPOCH)
        forward = offsets["itrf2014-to-etrs89"]
        self.assertIn("mean_enu_m", forward)
        east, north, up = forward["mean_enu_m"]
        # ~0.85 m horizontally, directed west-south-west, negligible vertical.
        self.assertAlmostEqual(east, -0.698, delta=0.02)
        self.assertAlmostEqual(north, -0.491, delta=0.02)
        self.assertAlmostEqual(up, 0.0, delta=0.01)
        self.assertAlmostEqual(forward["horizontal_m"], 0.854, delta=0.02)
        # The mirror operation is the exact inverse.
        back = offsets["etrs89-to-itrf2014"]["mean_enu_m"]
        for axis in range(3):
            self.assertAlmostEqual(back[axis], -forward["mean_enu_m"][axis],
                                   delta=1e-6)


if __name__ == "__main__":
    unittest.main()
