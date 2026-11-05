#!/usr/bin/env python3
"""WP-13.16: the Python Calibration file-I/O surface.

Pins that Calibration.load reads the JSON interchange format, that the loaded
object exposes the coefficients, and that Calibration.save writes a file the
loader accepts again. Warnings are returned as a list by save and raised
through the standard warnings module; a clean calibration produces none.

Run with plain unittest, no pytest.
"""

import inspect
import json
import os
import tempfile
import unittest
import warnings

import alphashape


def load_with_warning(path):
    """One hop above the binding, where stacklevel=2 reports the warning."""
    return alphashape.Calibration.load(path, alphashape.CalibrationFormat.OpenCvYaml)


class CalibrationIOTest(unittest.TestCase):
    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format": "alphashape-calibration",
                        "version": 1,
                        "type": "frame",
                        "width": 640,
                        "height": 480,
                        "f": 512.5,
                        "k1": -0.02,
                    },
                    handle,
                )

            loaded = alphashape.Calibration.load(
                source, alphashape.CalibrationFormat.Json
            )
            self.assertEqual(loaded.f, 512.5)
            self.assertEqual(loaded.width, 640)
            self.assertEqual(loaded.height, 480)
            self.assertEqual(loaded.k1, -0.02)

            target = os.path.join(directory, "out.json")
            warnings = loaded.save(target, alphashape.CalibrationFormat.Json)
            self.assertEqual(warnings, [])

            again = alphashape.Calibration.load(
                target, alphashape.CalibrationFormat.Json
            )
            self.assertEqual(again.f, 512.5)
            self.assertEqual(again.k1, -0.02)

    def test_warning_points_at_caller(self):
        yaml = (
            "%YAML:1.0\n---\n"
            "image_width: 1000\n"
            "image_height: 800\n"
            "camera_matrix: !!opencv-matrix\n"
            "   rows: 3\n   cols: 3\n   dt: f\n"
            "   data: [ 900.0, 0.0, 495.0, 0.0, 900.0, 405.0, 0.0, 0.0, 1.0 ]\n"
            "distortion_coefficients: !!opencv-matrix\n"
            "   rows: 1\n   cols: 8\n   dt: f\n"
            "   data: [ 0.1, 0.01, 0.001, 0.002, 0.0001, 0.5, 0.0, 0.0 ]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "in.yml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(yaml)
            frame = inspect.currentframe()
            self.assertIsNotNone(frame)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                line = frame.f_lineno + 1
                load_with_warning(path)
        self.assertTrue(caught)
        # stacklevel=2 steps over the Python wrapper, so the warning points at
        # the line in this module that called it, not at the bindings.
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, line)


if __name__ == "__main__":
    unittest.main()
