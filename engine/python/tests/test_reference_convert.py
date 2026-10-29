#!/usr/bin/env python3
"""Reference data conversion through the Python front end (WP-12.7).

What this file pins:

  * `chunk.convertReference(target_epsg, kind)` moves the camera reference
    positions through the CRS engine and relabels `camera_crs_epsg`/
    `marker_crs_epsg`; the chunk CRS and the poses are untouched.
  * `kind` selects which references convert, and an unknown kind or a
    non-positive EPSG code is refused.
  * The round trip returns every reference to its original position inside the
    documented 1e-9 degrees.

The scene is the 4x3 nadir block, whose EXIF carries exact GPS, so AddPhotos
alone puts real lon/lat/alt references on the cameras; no reconstruction is
needed and the whole file runs in about a second.

Run with plain unittest, no pytest.
"""

import os
import unittest

import alphashape


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
AERIAL = [
    os.path.join(DATA_DIR, "aerial_r{}c{}.jpg".format(row, col))
    for row in range(3)
    for col in range(4)
]

UTM_ZONE_32N = 32632
WGS84 = 4979


def georeferenced_chunk():
    """A chunk whose cameras carry their EXIF GPS reference."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(AERIAL))
    # AddPhotos writes the EXIF GPS into each camera's reference, as lon/lat/
    # alt. The reference CRS is unset until a slot says so; the test sets the
    # chunk CRS before converting, which both slots follow by default.
    if not all(camera.reference is not None for camera in chunk.cameras):
        raise AssertionError("the aerial fixtures carry no EXIF GPS")
    # The EXIF values are WGS 84 lon/lat/alt. Declaring it on the chunk makes
    # both reference slots follow it (they are unset until made explicit).
    chunk.crs_epsg = WGS84
    return doc, chunk


def references(chunk):
    return [camera.reference for camera in chunk.cameras]


class ConvertReferenceTest(unittest.TestCase):
    def setUp(self):
        # A fresh chunk per test: the conversions mutate the whole slot, so
        # sharing one across cases would make the result depend on test order.
        self.doc, self.chunk = georeferenced_chunk()

    def test_converts_to_utm_and_relabels(self):
        chunk = self.chunk
        before = references(chunk)
        chunk.convertReference(UTM_ZONE_32N, "cameras")
        # Zone 32N around this block: a plausible easting/northing pair.
        for lon, lat, _ in before:
            self.assertGreater(lon, 5.0)
            self.assertLess(lon, 15.0)
        for easting, northing, _ in references(chunk):
            self.assertGreater(easting, 300000.0)
            self.assertLess(easting, 700000.0)
            self.assertGreater(northing, 5000000.0)
            self.assertLess(northing, 6000000.0)
        self.assertEqual(chunk.camera_crs_epsg, UTM_ZONE_32N)
        # The chunk CRS is only a default for the slots; it is not rewritten.
        self.assertEqual(chunk.crs_epsg, WGS84)

    def test_round_trip_returns_to_the_original_degrees(self):
        chunk = self.chunk
        before = references(chunk)
        chunk.convertReference(UTM_ZONE_32N, "both")
        chunk.convertReference(WGS84, "both")
        for (lon, lat, _), (orig_lon, orig_lat, _) in zip(references(chunk), before):
            self.assertLess(abs(lon - orig_lon), 1e-9)
            self.assertLess(abs(lat - orig_lat), 1e-9)
        self.assertEqual(chunk.camera_crs_epsg, WGS84)

    def test_kind_selects_which_references_move(self):
        chunk = self.chunk
        # Make the camera slot explicit first: otherwise it would follow the
        # chunk CRS and the assertion below could not tell "untouched" from
        # "follows a different default".
        chunk.convertReference(UTM_ZONE_32N, "cameras")
        chunk.convertReference(WGS84, "cameras")
        before = references(chunk)
        chunk.convertReference(UTM_ZONE_32N, "markers")
        # No markers in this chunk, and the cameras must be untouched: a
        # markers-only conversion leaves the camera slot alone.
        self.assertEqual(references(chunk), before)
        self.assertEqual(chunk.marker_crs_epsg, UTM_ZONE_32N)
        self.assertEqual(chunk.camera_crs_epsg, WGS84)

    def test_rejects_bad_arguments(self):
        chunk = self.chunk
        with self.assertRaises(alphashape.ValueError):
            chunk.convertReference(UTM_ZONE_32N, "cameras_and_markers")
        with self.assertRaises(alphashape.ValueError):
            chunk.convertReference(0, "cameras")
        with self.assertRaises(alphashape.ValueError):
            chunk.convertReference(-32632, "cameras")


if __name__ == "__main__":
    unittest.main()
