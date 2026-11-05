#!/usr/bin/env python3
"""Camera reference attitude exposed to Python (WP-13.21).

The stored yaw/pitch/roll triple and its optional per-axis accuracy are read
through ``Camera.reference_orientation`` and
``Camera.reference_rotation_accuracy``. The first is populated by
ImportReference; the second is None unless a project sets it, which the C++
persistence test covers.
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


class TestCameraAttitude(unittest.TestCase):
    def make_document(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("attitude")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
        return doc, chunk

    def test_orientation_round_trip(self):
        doc, chunk = self.make_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reference.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("label,x,y,z,yaw,pitch,roll\n")
                handle.write("fixture_a,7.0,45.0,100.0,10.0,20.0,30.0\n")
            doc.run_task("ImportReference", path=path)

            camera = [c for c in chunk.cameras if c.label == "fixture_a"][0]
            self.assertEqual(camera.reference_orientation, (10.0, 20.0, 30.0))
            # No per-axis accuracy was imported: the chunk setting applies.
            self.assertIsNone(camera.reference_rotation_accuracy)

            project = os.path.join(tmp, "project.psx")
            doc.save(project)
            reloaded = alphashape.load_project(project)
            stored = reloaded.chunk(reloaded.active_chunk.id)
            again = [c for c in stored.cameras if c.label == "fixture_a"][0]
            self.assertEqual(again.reference_orientation, (10.0, 20.0, 30.0))
            self.assertIsNone(again.reference_rotation_accuracy)

    def test_rotation_accuracy_settable(self):
        doc, chunk = self.make_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reference.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("label,x,y,z,yaw,pitch,roll\n")
                handle.write("fixture_a,7.0,45.0,100.0,10.0,20.0,30.0\n")
            doc.run_task("ImportReference", path=path)

            camera = [c for c in chunk.cameras if c.label == "fixture_a"][0]
            chunk.set_camera_reference_rotation_accuracy(camera.id, (2.0, 0.5, 0.25))
            updated = [c for c in chunk.cameras if c.label == "fixture_a"][0]
            self.assertEqual(updated.reference_rotation_accuracy, (2.0, 0.5, 0.25))

            project = os.path.join(tmp, "project.psx")
            doc.save(project)
            reloaded = alphashape.load_project(project)
            stored = reloaded.chunk(reloaded.active_chunk.id)
            again = [c for c in stored.cameras if c.label == "fixture_a"][0]
            self.assertEqual(again.reference_rotation_accuracy, (2.0, 0.5, 0.25))

            stored.set_camera_reference_rotation_accuracy(again.id, None)
            cleared = [c for c in stored.cameras if c.label == "fixture_a"][0]
            self.assertIsNone(cleared.reference_rotation_accuracy)

        with self.assertRaises(alphashape.ValueError):
            chunk.set_camera_reference_rotation_accuracy(9999, (1.0, 1.0, 1.0))

    def test_absent_orientation_is_none(self):
        doc, chunk = self.make_document()
        for camera in chunk.cameras:
            self.assertIsNone(camera.reference_orientation)
            self.assertIsNone(camera.reference_rotation_accuracy)


if __name__ == "__main__":
    unittest.main()
