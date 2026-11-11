#!/usr/bin/env python3
"""Chunk reference settings through the Python front end (WP-13.1).

Pins the six per-chunk accuracy properties: each is the public Python API
spelling, reads back the value assigned, and refuses a non-positive value.
No image data or reconstruction is needed, so the file runs in well under a
second.

Run with plain unittest, no pytest.
"""

import os
import unittest

import alphashape

try:
    import numpy
except ImportError:  # the module itself does not depend on numpy
    numpy = None

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
CAM_A1 = os.path.join(DATA_DIR, "cam_a1.jpg")


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


class SensorAntennaTest(unittest.TestCase):
    """The antenna lever-arm setter accepts an array (pybind11's Eigen caster).

    Before the caster was included, ``set_sensor_antenna`` compiled as a
    generic class caster and rejected every Python value: lists and numpy
    arrays alike. The GCP protocol depends on this call, so pin it.
    """

    @unittest.skipUnless(numpy is not None, "numpy is required by the Eigen argument")
    def test_antenna_offset_accepts_a_numpy_vector(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[CAM_A1])
        self.assertEqual(len(chunk.sensors), 1)

        offset = numpy.array([0.0, -0.2, -0.23])
        chunk.set_sensor_antenna(0, offset)
        stored = chunk.sensors[0].antenna_offset
        for got, want in zip(stored, offset):
            self.assertAlmostEqual(got, float(want), places=12)


if __name__ == "__main__":
    unittest.main()
