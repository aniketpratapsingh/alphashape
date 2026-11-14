#!/usr/bin/env python3
"""Event-mark and capture-timing parsers used by the real-block GCP protocol.

The real-block diagnostics compare each image's capture time with the RTK
module's per-exposure event mark. The exposure-mark file is a whitespace
formatted log whose documented columns are::

    index  gps_time_of_week  [gps_week]  <solution tokens>
    lat,Lat  lon,Lon  height,Ellh
    sigma_lat, sigma_lon, sigma_height  flag,Quality

and the JPEG comment carries two integer counters, ``timestamp`` (shutter
command, milliseconds) and ``exppts`` (exposure event, microseconds), on the
module's own clock. These tests pin the parsers only: nothing here needs a
chunk, so the script is loaded by path and the parsing functions are exercised
directly.

Plain unittest, no pytest.
"""

import importlib.util
import os
import pathlib
import tempfile
import unittest


def load_script():
    path = (pathlib.Path(__file__).resolve().parents[2] /
            "tests" / "run_gcp_validation.py")
    spec = importlib.util.spec_from_file_location("run_gcp_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()

# Two lines in the shape the dataset's RTK module writes. The first matches
# the real block's first exposure (MAX_0002); the second is a fixed solution
# with a larger height deviation.
MRK_TEXT = """\
2\t296288.242869\t[2211]\t3,N\t0,E\t75,V\t47.6436873208,Lat\t16.4759271684,Lon\t512.998,Ellh\t0.014391, 0.013698, 0.029694\t50,Q
3\t296289.777625\t[2211]\t5,N\t1,E\t75,V\t47.6436479170,Lat\t16.4758945272,Lon\t512.996,Ellh\t0.014268, 0.013609, 0.029783\t50,Q
"""


class TestParseMrk(unittest.TestCase):
    def _write(self, directory, text=MRK_TEXT):
        path = os.path.join(directory, "events.mrk")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_reads_position_sigma_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            records = SCRIPT.parse_mrk(self._write(directory))
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first["index"], 2)
        self.assertAlmostEqual(first["time_s"], 296288.242869)
        self.assertEqual(first["week"], 2211)
        self.assertAlmostEqual(first["lat"], 47.6436873208)
        self.assertAlmostEqual(first["lon"], 16.4759271684)
        self.assertAlmostEqual(first["height_m"], 512.998)
        # The file's deviations are latitude, longitude, height (not the
        # XMP order), so the parser keeps them under those names.
        self.assertAlmostEqual(first["sigma_lat_m"], 0.014391)
        self.assertAlmostEqual(first["sigma_lon_m"], 0.013698)
        self.assertAlmostEqual(first["sigma_hgt_m"], 0.029694)
        self.assertEqual(first["fix_flag"], 50)
        self.assertEqual(first["quality"], "Q")

    def test_skips_malformed_lines_and_keeps_the_rest(self):
        text = MRK_TEXT + "not a record\n" + \
            "4\t296291.0\t[2211]\t5,N\t0,E\t75,V\t47.6,Lat\t16.4,Lon\n"
        with tempfile.TemporaryDirectory() as directory:
            records = SCRIPT.parse_mrk(self._write(directory, text))
        self.assertEqual([record["index"] for record in records], [2, 3])

    def test_empty_file_is_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(SCRIPT.parse_mrk(self._write(directory, "")), [])


class TestCaptureTiming(unittest.TestCase):
    def test_reads_the_two_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frame.jpg")
            with open(path, "wb") as handle:
                handle.write(b"\xff\xd8 prefix timestamp=749607\n"
                             b"exppts=749792242\n trailing")
            timing = SCRIPT.read_capture_timing(path)
        self.assertEqual(timing["trigger_us"], 749607000)
        self.assertEqual(timing["exposure_us"], 749792242)

    def test_absent_comment_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frame.jpg")
            with open(path, "wb") as handle:
                handle.write(b"\xff\xd8 no timing here")
            self.assertIsNone(SCRIPT.read_capture_timing(path))


class TestAnalyseTiming(unittest.TestCase):
    def test_event_matches_exposure_and_trigger_precedes_it(self):
        # Two exposures 1.5 s apart, 5 m of track: a 3.33 m/s block speed.
        records = [
            {"index": 2, "time_s": 1000.0, "week": 2211,
             "lat": 47.0, "lon": 16.0, "height_m": 500.0,
             "sigma_lat_m": 0.014, "sigma_lon_m": 0.013,
             "sigma_hgt_m": 0.029, "fix_flag": 50, "quality": "Q"},
            {"index": 3, "time_s": 1001.5, "week": 2211,
             "lat": 47.0 + 5.0 / 111320.0, "lon": 16.0, "height_m": 500.0,
             "sigma_lat_m": 0.014, "sigma_lon_m": 0.013,
             "sigma_hgt_m": 0.029, "fix_flag": 50, "quality": "Q"},
        ]
        reference_rows = [
            {"label": "MAX_0002",
             "reference": (records[0]["lon"], records[0]["lat"],
                           records[0]["height_m"])},
            {"label": "MAX_0003",
             "reference": (records[1]["lon"], records[1]["lat"],
                           records[1]["height_m"])},
        ]
        with tempfile.TemporaryDirectory() as directory:
            images = []
            for index, label in ((2, "MAX_0002"), (3, "MAX_0003")):
                path = os.path.join(directory, label + ".JPG")
                exposure_us = 700000000 + index * 1500000
                trigger_us = exposure_us - 185000
                comment = "timestamp={}\nexppts={}\n".format(
                    trigger_us // 1000, exposure_us)
                with open(path, "wb") as handle:
                    handle.write(b"\xff\xd8 " + comment.encode("ascii"))
                images.append(path)
            report = SCRIPT.analyse_timing(reference_rows, images, records)

        self.assertEqual(report["matched_cameras"], 2)
        self.assertEqual(report["images"], 2)
        # The event marks and the exposure counters share a time base: after
        # removing the constant clock offset, the disagreement is sub-ms.
        self.assertLess(report["event_vs_exposure_max_abs_s"], 0.001)
        # The shutter command precedes the exposure by the same kind of delay
        # the dataset shows; the event mark is stamped at the exposure, so the
        # recorded position is not late.
        self.assertGreater(report["trigger_to_exposure_median_s"], 0.0)
        self.assertAlmostEqual(report["block_speed_m_s"], 5.0 / 1.5, places=2)
        self.assertEqual(report["fix_status_counts"], {"50": 2})


class TestCameraAccuracyMode(unittest.TestCase):
    def test_per_image_and_scalar_forms(self):
        self.assertEqual(SCRIPT.parse_camera_accuracy("per-image"),
                         SCRIPT.CAMERA_ACCURACY_PER_IMAGE)
        self.assertEqual(SCRIPT.parse_camera_accuracy("0.05"), 0.05)
        for bad in ("0", "-1", "nan", "not-a-number"):
            with self.assertRaises(ValueError):
                SCRIPT.parse_camera_accuracy(bad)


if __name__ == "__main__":
    unittest.main()
