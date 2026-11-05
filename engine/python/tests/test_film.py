#!/usr/bin/env python3
"""WP-13.18b: the film-normalised geometry on the Python Sensor view.

Pins that set_sensor_film stores the resolution and extent and that the
Sensor exposes them back. Plain unittest, like the other Python tests.
"""

import os
import unittest

import alphashape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
CAM = os.path.join(DATA_DIR, "cam_a1.jpg")


class FilmGeometryPropertiesTest(unittest.TestCase):
    def test_set_and_read_normalised_geometry(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("film")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[CAM])
        chunk.set_sensor_film(
            0,
            True,
            [("a", -10.0, 10.0), ("b", 10.0, 10.0), ("c", 10.0, -10.0)],
            transform="affine",
            detector="circle",
            corners=True,
            sides=False,
            resolution_mm_per_px=0.05,
            film_size_mm=(0.0, 0.0),
        )
        sensor = chunk.sensors[0]
        self.assertTrue(sensor.film_enabled)
        self.assertAlmostEqual(sensor.film_resolution_mm_per_px, 0.05)
        self.assertEqual(tuple(sensor.film_size_mm), (0.0, 0.0))
        self.assertEqual(len(sensor.fiducials), 3)

    def test_extent_is_stored(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("film")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[CAM])
        chunk.set_sensor_film(
            0,
            True,
            [("a", -10.0, 10.0), ("b", 10.0, 10.0), ("c", 10.0, -10.0)],
            resolution_mm_per_px=0.02,
            film_size_mm=(120.0, 130.0),
        )
        size = tuple(chunk.sensors[0].film_size_mm)
        self.assertEqual(size, (120.0, 130.0))

    def test_negative_resolution_is_rejected(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("film")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[CAM])
        with self.assertRaises(alphashape.ValueError):
            chunk.set_sensor_film(
                0, True, [("a", 0.0, 0.0)], resolution_mm_per_px=-1.0
            )


if __name__ == "__main__":
    unittest.main()
