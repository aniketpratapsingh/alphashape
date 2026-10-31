#!/usr/bin/env python3
"""Chunk reference settings through the Python front end (WP-13.1).

Pins the six per-chunk accuracy properties: each is the public Python API
spelling, reads back the value assigned, and refuses a non-positive value.
No image data or reconstruction is needed, so the file runs in well under a
second.

Run with plain unittest, no pytest.
"""

import unittest

import alphashape


class ReferenceSettingsTest(unittest.TestCase):
    def setUp(self):
        self.doc = alphashape.Document()
        self.chunk = self.doc.add_chunk("Chunk 1")

    def test_defaults_match_our_model(self):
        expected = {
            "camera_location_accuracy": 10.0,
            "camera_rotation_accuracy": 10.0,
            "marker_location_accuracy": 0.005,
            "marker_projection_accuracy": 0.5,
            "scalebar_accuracy": 0.001,
            "tiepoint_accuracy": 1.0,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(getattr(self.chunk, name), value, places=12)

    def test_properties_round_trip(self):
        changed = {
            "camera_location_accuracy": 0.25,
            "camera_rotation_accuracy": 1.5,
            "marker_location_accuracy": 0.01,
            "marker_projection_accuracy": 0.75,
            "scalebar_accuracy": 0.002,
            "tiepoint_accuracy": 2.0,
        }
        for name, value in changed.items():
            setattr(self.chunk, name, value)
        for name, value in changed.items():
            self.assertAlmostEqual(getattr(self.chunk, name), value, places=12)

    def test_rejects_non_positive_accuracy(self):
        for name in ("camera_location_accuracy", "marker_location_accuracy",
                     "scalebar_accuracy"):
            with self.assertRaises(Exception):
                setattr(self.chunk, name, 0.0)
            with self.assertRaises(Exception):
                setattr(self.chunk, name, -1.0)


if __name__ == "__main__":
    unittest.main()
