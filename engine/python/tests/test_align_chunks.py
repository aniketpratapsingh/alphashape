#!/usr/bin/env python3
"""Marker-based chunk alignment (M11 WP-11.7) through the Python front end.

What this file pins:

  * Chunk.markers is a list of {id, label, position, reference} dicts, and
    Chunk.set_marker_position places or moves a marker by label;
  * Document.align_chunks aligns the ACTIVE chunk onto a reference chunk from
    markers with matching labels and records the applied 4x4 in both chunks'
    meta under "AlignChunks/<other id>/matrix";
  * method="reference" is an identity on two chunks with the same CRS and is
    refused when they are not georeferenced;
  * the task's parameters reach the project file like every other task's.

Plain unittest, no pytest.
"""

import math
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

# Six non-collinear physical points; independent of the camera geometry.
POINTS = [
    (0.0, 0.0, 0.0),
    (30.0, 0.0, 5.0),
    (0.0, 25.0, 10.0),
    (20.0, 20.0, 3.0),
    (-25.0, 10.0, -4.0),
    (-10.0, -20.0, 7.0),
]

# The known similarity the active chunk is moved by: s = 1.7, 15 deg about z,
# t = (100, -40, 2). Marker positions in the ACTIVE chunk are its PRE-images.
SCALE = 1.7
ANGLE = math.radians(15.0)
TX, TY, TZ = 100.0, -40.0, 2.0


def forward(point):
    x, y, z = point
    return (
        SCALE * (math.cos(ANGLE) * x - math.sin(ANGLE) * y) + TX,
        SCALE * (math.sin(ANGLE) * x + math.cos(ANGLE) * y) + TY,
        SCALE * z + TZ,
    )


def two_chunk_document(second_on_truth=True):
    """Two chunks of the same frames with matching marker labels.

    The active chunk holds the marker pre-images; the reference chunk holds
    their images under the known similarity (or the same positions, for the
    reference-method cases).
    """
    doc = alphashape.Document()
    active = doc.add_chunk("Active")
    doc.active_chunk = active
    doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
    reference = doc.add_chunk("Reference")
    # add_chunk() does not change the active chunk, and the task layer resolves
    # its target that way, so point at the new chunk for the import.
    doc.active_chunk = reference
    doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
    for i, point in enumerate(POINTS):
        label = "M{}".format(i + 1)
        active.set_marker_position(label, *point)
        reference.set_marker_position(label, *(forward(point) if second_on_truth else point))
    doc.active_chunk = active
    return doc, active, reference


class TestMarkerSurface(unittest.TestCase):
    def test_set_marker_position_creates_then_moves(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        self.assertEqual(chunk.markers, [])

        first = chunk.set_marker_position("GCP 1", 1.0, 2.0, 3.0)
        self.assertEqual(chunk.markers[0]["id"], first)
        self.assertEqual(chunk.markers[0]["label"], "GCP 1")
        self.assertEqual(chunk.markers[0]["position"], (1.0, 2.0, 3.0))
        self.assertIsNone(chunk.markers[0]["reference"])

        # Same label: moved, not duplicated.
        chunk.set_marker_position("GCP 1", 4.0, 5.0, 6.0)
        self.assertEqual(len(chunk.markers), 1)
        self.assertEqual(chunk.markers[0]["position"], (4.0, 5.0, 6.0))


class TestAlignChunks(unittest.TestCase):
    def test_recovers_the_known_similarity(self):
        doc, active, reference = two_chunk_document()
        # The imported chunks carry no poses yet, so the expectation is the
        # mapped pre-alignment centre: C' = s R C + t.
        before = [camera.center for camera in active.cameras]
        doc.align_chunks(reference)

        key = "AlignChunks/{}/matrix".format(reference.id)
        values = [float(text) for text in active.meta[key].split()]
        self.assertEqual(len(values), 16)
        # Row-major: row 0 is [s cos, -s sin, 0, tx].
        self.assertAlmostEqual(values[0], SCALE * math.cos(ANGLE), places=9)
        self.assertAlmostEqual(values[1], -SCALE * math.sin(ANGLE), places=9)
        self.assertAlmostEqual(values[3], TX, places=6)
        self.assertAlmostEqual(values[7], TY, places=6)
        self.assertAlmostEqual(values[11], TZ, places=6)
        self.assertAlmostEqual(values[10], SCALE, places=9)

        # The reference chunk records the inverse under the active chunk's id.
        back = "AlignChunks/{}/matrix".format(active.id)
        self.assertIn(back, reference.meta)

        # Every active camera centre now sits where the similarity maps it:
        # the same rule the reference chunk's cameras follow.
        for centre, original in zip((c.center for c in active.cameras), before):
            expected = forward(original)
            for axis in range(3):
                self.assertAlmostEqual(centre[axis], expected[axis], places=6)

    def test_requires_three_markers(self):
        doc, active, reference = two_chunk_document()
        doc.align_chunks(reference)
        with self.assertRaises(alphashape.Error):
            doc.run_task("AlignChunks", reference_chunk_id=reference.id,
                         min_common_markers=7)

    def test_unknown_method_is_rejected(self):
        doc, _, reference = two_chunk_document()
        with self.assertRaises(alphashape.ValueError):
            doc.align_chunks(reference, method="point_cloud")

    def test_reference_method_is_an_identity_on_a_shared_crs(self):
        doc, active, reference = two_chunk_document(second_on_truth=False)
        active.crs_epsg = 4326
        reference.crs_epsg = 4326
        before = [(camera.center) for camera in active.cameras]
        doc.align_chunks(reference, method="reference")
        for centre, expected in zip((c.center for c in active.cameras), before):
            self.assertEqual(centre, expected)
        self.assertIn("AlignChunks/{}/matrix".format(reference.id), active.meta)

    def test_reference_method_requires_georeferenced_chunks(self):
        doc, active, reference = two_chunk_document(second_on_truth=False)
        active.crs_epsg = 0
        with self.assertRaises(alphashape.Error):
            doc.align_chunks(reference, method="reference")


class TestAlignChunksPersistence(unittest.TestCase):
    """The alignment record and the moved geometry survive a save/load."""

    def test_alignment_survives_save_and_load(self):
        doc, active, reference = two_chunk_document()
        doc.align_chunks(reference)
        key = "AlignChunks/{}/matrix".format(reference.id)
        recorded = active.meta[key]
        centres = [camera.center for camera in active.cameras]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aligned.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)

        loaded = reloaded.chunk(active.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.meta[key], recorded)
        for centre, expected in zip((c["position"] for c in loaded.markers),
                                    (c["position"] for c in active.markers)):
            self.assertEqual(centre, expected)
        for camera, expected in zip(loaded.cameras, centres):
            for axis in range(3):
                self.assertAlmostEqual(camera.center[axis], expected[axis], places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
