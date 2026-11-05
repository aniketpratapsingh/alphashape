#!/usr/bin/env python3
"""WP-13.16: the Python Calibration file-I/O surface.

Pins that Calibration.load reads the JSON interchange format, that the loaded
object exposes the coefficients, and that Calibration.save writes a file the
loader accepts again. Warnings are returned as a list by save and raised
through the standard warnings module; a clean calibration produces none.

Run with plain unittest, no pytest.
"""

import json
import os
import tempfile
import unittest

import alphashape


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

            again = alphashape.Calibration.load(target)
            self.assertEqual(again.f, 512.5)
            self.assertEqual(again.k1, -0.02)


if __name__ == "__main__":
    unittest.main()
