#!/usr/bin/env python3
"""UpdateTransform binding (WP-13.12).

`Chunk.updateTransform()` runs the UpdateTransform task through the same
`run_task` path the other chunk tasks use. These tests pin the binding and the
refusal contract: without three usable references the task records
`update_transform.applied = 0` with a reason and leaves the geometry alone.

A successful fit needs camera or marker references, which the Python API has no
setter for (the reference table is read-only here); the success path is covered
by the C++ update_transform tests.

Plain unittest, no pytest.
"""

import os
import unittest

import alphashape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
FIXTURE_A = os.path.join(DATA_DIR, "fixture_a.pgm")
FIXTURE_B = os.path.join(DATA_DIR, "fixture_b.pgm")


def document_with_photos():
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
    return doc, chunk


class TestUpdateTransform(unittest.TestCase):
    def test_refusal_records_reason_and_leaves_geometry(self):
        _, chunk = document_with_photos()
        transform_before = chunk.transform
        poses_before = [camera.pose for camera in chunk.cameras]

        chunk.updateTransform()

        self.assertEqual(chunk.meta["update_transform.applied"], "0")
        self.assertIn("reference", chunk.meta["update_transform.reason"])
        self.assertEqual(chunk.transform, transform_before)
        self.assertEqual([camera.pose for camera in chunk.cameras], poses_before)

    def test_markers_without_references_still_refuse(self):
        _, chunk = document_with_photos()
        for index in range(3):
            chunk.addMarker(label="GCP{}".format(index),
                            position=(float(index), float(-index), 1.0))
        positions_before = [marker["position"] for marker in chunk.markers]

        chunk.updateTransform()

        self.assertEqual(chunk.meta["update_transform.applied"], "0")
        self.assertIn("reference", chunk.meta["update_transform.reason"])
        self.assertEqual([marker["position"] for marker in chunk.markers],
                         positions_before)


if __name__ == "__main__":
    unittest.main()
