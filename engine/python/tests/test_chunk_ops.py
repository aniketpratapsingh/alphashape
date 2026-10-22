#!/usr/bin/env python3
"""Python surface for the chunk operations (M11 WP-11.6).

Document.duplicate_chunk / merge_chunks / append_project and the chunk-subset
form of Document.save. Plain unittest, like the other Python tests.
"""

import os
import tempfile
import unittest

import alphashape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
CAM_A1 = os.path.join(DATA_DIR, "cam_a1.jpg")
CAM_A2 = os.path.join(DATA_DIR, "cam_a2.jpg")
CAM_B1 = os.path.join(DATA_DIR, "cam_b1.jpg")


def imported_document(*paths):
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(paths))
    return doc, chunk


class TestDuplicateChunk(unittest.TestCase):
    def test_duplicate_is_independent_and_remaps_references(self):
        doc, source = imported_document(CAM_A1, CAM_A2, CAM_B1)
        copy = doc.duplicate_chunk(source, "copy")
        self.assertEqual(copy.id, 1)
        self.assertEqual(copy.label, "copy")
        self.assertEqual(copy.camera_count, 3)
        self.assertEqual(len(copy.sensors), len(source.sensors))
        # The copy owns its cameras and sensors; the source is untouched.
        self.assertEqual(doc.chunk(source.id).camera_count, 3)

        # Default label: "<source label> copy".
        again = doc.duplicate_chunk()
        self.assertEqual(again.label, "copy copy")

    def test_duplicate_of_missing_chunk_raises(self):
        doc, _ = imported_document(CAM_A1)
        with self.assertRaises(alphashape.ValueError):
            doc.duplicate_chunk(424242)


class TestMergeChunks(unittest.TestCase):
    def test_merge_concatenates_cameras(self):
        doc, first = imported_document(CAM_A1)
        second = doc.add_chunk("Second")
        doc.active_chunk = second
        doc.run_task("AddPhotos", filenames=[CAM_B1])
        doc.active_chunk = first

        merged = doc.merge_chunks([first.id, second.id], label="together")
        self.assertEqual(merged.label, "together")
        self.assertEqual(merged.camera_count, 2)
        self.assertEqual(len(merged.sensors), 2)
        # Both source chunks are still present and unchanged.
        self.assertEqual(len(doc.chunk(first.id).cameras), 1)
        self.assertEqual(len(doc.chunk(second.id).cameras), 1)

    def test_merge_needs_two_chunks(self):
        doc, first = imported_document(CAM_A1)
        with self.assertRaises(alphashape.ValueError):
            doc.merge_chunks([first.id])


class TestAppendProjectAndSubsetSave(unittest.TestCase):
    def test_subset_save_round_trip(self):
        doc, first = imported_document(CAM_A1, CAM_A2)
        second = doc.add_chunk("Second")
        doc.active_chunk = second
        doc.run_task("AddPhotos", filenames=[CAM_B1])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subset.psx")
            doc.save(path, chunks=[first.id])
            loaded = alphashape.load_project(path)
            self.assertEqual(len(loaded.chunk_ids), 1)
            self.assertEqual(loaded.active_chunk.id, first.id)

    def test_append_project_offsets_ids(self):
        source, source_chunk = imported_document(CAM_A1, CAM_A2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "source.psx")
            source.save(path)

            dest, dest_chunk = imported_document(CAM_B1)
            dest.append_project(path)
            self.assertEqual(len(dest.chunk_ids), 2)
            appended = dest.active_chunk
            self.assertEqual(appended.id, 1)
            self.assertEqual(appended.camera_count, 2)
            # The destination chunk keeps its single camera.
            self.assertEqual(len(dest.chunk(dest_chunk.id).cameras), 1)

    def test_append_unknown_chunk_id_raises(self):
        source, _ = imported_document(CAM_A1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "source.psx")
            source.save(path)
            dest, _ = imported_document(CAM_B1)
            with self.assertRaises(alphashape.ValueError):
                dest.append_project(path, chunk_ids=[99])
            self.assertEqual(len(dest.chunk_ids), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
