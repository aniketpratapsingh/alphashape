#!/usr/bin/env python3
"""Manual marker placement and marker import/export (WP-13.4).

What this file pins:

  * Chunk.addMarker creates a marker and returns its dict;
  * Chunk.setMarkerProjection accepts a label (creating the marker when absent)
    and an integer id (which must name an existing marker), and the two paths
    never shadow each other even when a marker is labelled with a number;
  * Chunk.removeMarkerProjection(delete_marker=True) removes a marker with its
    last projection;
  * importMarkers/exportMarkers round trip through both codecs.

Plain unittest, no pytest.
"""

import os
import tempfile
import unittest

import alphashape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
FIXTURE_A = os.path.join(DATA_DIR, "fixture_a.pgm")
FIXTURE_B = os.path.join(DATA_DIR, "fixture_b.pgm")


def document_with_photos():
    """A document whose active chunk has two cameras with labels."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
    return doc, chunk


class TestMarkerPlacementApi(unittest.TestCase):
    def test_add_place_remove(self):
        _, chunk = document_with_photos()
        marker = chunk.addMarker(label="GCP1", position=(1.0, 2.0, 3.0))
        self.assertEqual(marker["label"], "GCP1")
        self.assertEqual(marker["position"], (1.0, 2.0, 3.0))

        camera = chunk.cameras[0].label
        # Label path: the marker exists, so this only adds the projection.
        chunk.setMarkerProjection("GCP1", camera, 10.0, 20.0)
        # Id path: replaces the same projection.
        chunk.setMarkerProjection(marker["id"], camera, 11.0, 21.0)
        self.assertEqual(len(chunk.markers), 1)

        # A string that names no marker creates one; an id that names none is
        # an error and creates nothing.
        chunk.setMarkerProjection("NEW", camera, 1.0, 2.0)
        self.assertEqual(len(chunk.markers), 2)
        with self.assertRaises(alphashape.ValueError):
            chunk.setMarkerProjection(9999, camera, 0.0, 0.0)
        self.assertEqual(len(chunk.markers), 2)

        chunk.removeMarkerProjection("GCP1", camera, delete_marker=True)
        self.assertEqual([m["label"] for m in chunk.markers], ["NEW"])

    def test_numeric_label_does_not_shadow_id(self):
        _, chunk = document_with_photos()
        numeric = chunk.addMarker(label="5")
        other = chunk.addMarker(label="other")
        camera = chunk.cameras[0].label
        self.assertNotEqual(numeric["id"], other["id"])

        chunk.setMarkerProjection(other["id"], camera, 7.0, 8.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "markers.csv")
            chunk.exportMarkers(path=path)
            with open(path, encoding="utf-8") as handle:
                rows = handle.read().splitlines()[1:]
        # Only the id-addressed marker was written, never the one labelled "5".
        fields = [row.split(",") for row in rows]
        self.assertEqual([field[0] for field in fields], ["other"])
        self.assertEqual(fields[0][1], camera)
        self.assertEqual(float(fields[0][2]), 7.0)
        self.assertEqual(float(fields[0][3]), 8.0)

    def test_import_export_round_trip(self):
        _, source = document_with_photos()
        cameras = [camera.label for camera in source.cameras]
        source.addMarker(label="GCP1", position=(1.25, -2.5, 3.75))
        for index, camera in enumerate(cameras):
            source.setMarkerProjection("GCP1", camera,
                                       10.5 + index, 20.25 + index)

        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".csv", ".xml"):
                _, destination = document_with_photos()
                path = os.path.join(directory, "markers" + suffix)
                source.exportMarkers(path=path)
                destination.importMarkers(path=path)
                labels = [m["label"] for m in destination.markers]
                self.assertEqual(labels, ["GCP1"])
                if suffix == ".xml":
                    self.assertEqual(destination.markers[0]["position"],
                                     (1.25, -2.5, 3.75))
                else:
                    # CSV carries projections only, not an estimated position.
                    self.assertIsNone(destination.markers[0]["position"])

                # Re-exporting the imported chunk reproduces the file: the
                # projections (label, camera, pixel) survived the round trip.
                again = os.path.join(directory, "again" + suffix)
                destination.exportMarkers(path=again)
                with open(path, encoding="utf-8") as first, \
                        open(again, encoding="utf-8") as second:
                    self.assertEqual(first.read(), second.read())


if __name__ == "__main__":
    unittest.main()
