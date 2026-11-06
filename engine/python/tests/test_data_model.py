#!/usr/bin/env python3
"""The completed data model through the Python front end (WP-11.11).

The public scripting names are exercised directly: Document.chunks /
addChunk / remove / append / save / merge_chunks / undo / redo, every Chunk
container, the Marker / MarkerGroup / Scalebar / Asset views and the proxy
guarantee (a held element stays correct across vector mutations).

Run with plain unittest, no pytest.
"""

import os
import tempfile
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


def empty_document():
    doc = alphashape.Document()
    chunk = doc.addChunk("Chunk 1")
    return doc, chunk


class TestContainers(unittest.TestCase):
    def test_all_containers_visible(self):
        doc, chunk = empty_document()
        for name in ("cameras", "sensors", "frames", "markers",
                     "marker_groups", "scalebars", "point_clouds", "models",
                     "depth_maps", "depth_maps_sets", "elevations",
                     "orthomosaics", "shapes", "trajectories", "masks",
                     "tiled_models"):
            self.assertIsInstance(getattr(chunk, name), list, name)
        self.assertIsNone(chunk.panorama)
        self.assertIsInstance(chunk.transform, list)
        self.assertEqual(len(chunk.transform), 4)
        self.assertEqual(set(chunk.region), {"center", "size"})
        self.assertIsInstance(chunk.meta, dict)
        self.assertIsInstance(chunk.key, str)
        self.assertEqual(chunk.key, str(chunk.id))
        self.assertTrue(chunk.enabled)
        # Document-level views.
        self.assertIsInstance(doc.chunks, dict)
        self.assertEqual(list(doc.chunks), [chunk.key])
        self.assertEqual(doc.chunks[chunk.key].label, "Chunk 1")
        self.assertEqual(doc.chunk_ids, [chunk.id])


class TestMultipleAssets(unittest.TestCase):
    """Several assets of one kind, and the Asset view they hand out."""

    @classmethod
    def setUpClass(cls):
        cls.doc = alphashape.Document()
        cls.chunk = cls.doc.addChunk("Chunk 1")
        cls.doc.active_chunk = cls.chunk
        cls.doc.run_task("AddPhotos", filenames=list(AERIAL))
        cls.doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        cls.doc.run_task("AlignCameras")
        cls.doc.run_task("BuildDepthMaps", downscale=8)
        cls.chunk.buildDem(source_data="depth_maps", interpolation="enabled",
                           smoothing="none", label="dem a")
        cls.chunk.buildDem(source_data="depth_maps", interpolation="enabled",
                           smoothing="none", label="dem b")

    def test_multiple_assets_visible(self):
        sets = self.chunk.depth_maps_sets
        dems = self.chunk.elevations
        self.assertGreaterEqual(len(sets), 1)
        self.assertEqual([asset.label for asset in dems], ["dem a", "dem b"])
        ids = [asset.id for asset in dems]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual([asset.active for asset in dems], [False, True])
        self.assertEqual(dems[0].group, None)
        self.assertTrue(all(asset.enabled for asset in dems))
        # The dict protocol of the old asset lists still works.
        self.assertEqual(dems[-1]["label"], "dem b")
        self.assertEqual(dems[-1]["key"], dems[-1].id)
        self.assertEqual(dict(dems[-1])["label"], "dem b")
        # All depth maps across every set are flattened into `depth_maps`;
        # with a single set the active maps are the identical full entries.
        all_maps = self.chunk.depth_maps
        active_maps = self.chunk.active_depth_maps
        self.assertEqual(len(all_maps), len(active_maps))
        for all_map, active_map in zip(all_maps, active_maps):
            self.assertEqual(set(all_map), set(active_map))
            self.assertEqual(set(all_map),
                             {"camera_id", "width", "height", "downscale",
                              "depth", "normals", "confidence"})
            self.assertEqual(all_map, active_map)
        self.assertIsNotNone(self.chunk.elevation)
        self.assertEqual(self.chunk.elevation.source, "depth_maps")

    def test_camera_rotation_accuracy_validation(self):
        camera_id = self.chunk.cameras[0].id
        for bad in ((float("nan"), 1.0, 1.0),
                    (1.0, float("inf"), 1.0),
                    (-1.0, 1.0, 1.0),
                    (1.0, 2.0)):
            with self.assertRaises(Exception):
                self.chunk.set_camera_reference_rotation_accuracy(
                    camera_id, bad)
        # The command is one undo step and restores the previous absence.
        self.chunk.set_camera_reference_rotation_accuracy(camera_id, (1.0, 2.0, 3.0))
        self.assertEqual(
            self.chunk.cameras[0].reference_rotation_accuracy, (1.0, 2.0, 3.0))
        self.doc.undo()
        self.assertIsNone(self.chunk.cameras[0].reference_rotation_accuracy)


class TestMarkerCrudAndUndo(unittest.TestCase):
    def test_marker_crud_and_undo(self):
        doc, chunk = empty_document()
        marker = chunk.addMarker(label="GCP1", position=(1.0, 2.0, 3.0))
        marker_id = marker.id
        self.assertEqual(marker["label"], "GCP1")
        self.assertTrue(doc.can_undo)

        marker.label = "renamed"
        self.assertEqual(chunk.markers[0].label, "renamed")
        doc.undo()
        self.assertEqual(chunk.markers[0].label, "GCP1")
        doc.redo()
        self.assertEqual(chunk.markers[0].label, "renamed")

        marker.reference = (10.0, 20.0, 30.0)
        self.assertEqual(chunk.markers[0].reference, (10.0, 20.0, 30.0))
        doc.undo()
        self.assertIsNone(chunk.markers[0].reference)
        doc.redo()
        self.assertEqual(chunk.markers[0].reference, (10.0, 20.0, 30.0))

        chunk.removeMarker(marker_id)
        self.assertEqual(chunk.markers, [])
        doc.undo()
        self.assertEqual(len(chunk.markers), 1)
        restored = chunk.markers[0]
        self.assertEqual(
            (restored.id, restored.label, restored.reference, restored.position),
            (marker_id, "renamed", (10.0, 20.0, 30.0), (1.0, 2.0, 3.0)),
        )

        # Undo the reference, the rename and the add itself, one step each.
        doc.undo()
        self.assertIsNone(chunk.markers[0].reference)
        doc.undo()
        self.assertEqual(chunk.markers[0].label, "GCP1")
        doc.undo()
        self.assertEqual(chunk.markers, [])
        # Only the chunk creation (an edit of its own) remains on the stack.
        self.assertTrue(doc.can_undo)

        # Redo replays the same four steps and ends with the removal.
        for _ in range(4):
            doc.redo()
        self.assertEqual(chunk.markers, [])

    def test_proxy_survives_mutation(self):
        doc, chunk = empty_document()
        held = chunk.addMarker(label="held", position=(4.0, 5.0, 6.0))
        held_id = held.id
        for index in range(100):
            chunk.addMarker(label="m{}".format(index), position=(0.0, 0.0, 0.0))
        self.assertEqual(len(chunk.markers), 101)
        # The held proxy refers to its element by id and re-resolves, so the
        # 100 insertions (and the vector reallocations they caused) are safe.
        self.assertEqual(held.id, held_id)
        self.assertEqual(held.label, "held")
        self.assertEqual(held.position, (4.0, 5.0, 6.0))
        held.label = "still mine"
        self.assertEqual(chunk.markers[0].label, "still mine")

    def test_marker_mapping_protocol(self):
        doc, chunk = empty_document()
        marker = chunk.addMarker(label="mapped", position=(1.0, 2.0, 3.0))
        self.assertEqual(dict(marker)["label"], "mapped")
        self.assertEqual(marker.get("missing", "fallback"), "fallback")
        self.assertEqual(dict(marker.items())["id"], marker.id)
        self.assertEqual(sorted(marker), sorted(marker.keys()))
        self.assertIn("valid", marker)

    def test_set_marker_position_is_one_undo(self):
        doc, chunk = empty_document()
        marker_id = chunk.set_marker_position("placed", 1.0, 2.0, 3.0)
        self.assertEqual(len(chunk.markers), 1)
        self.assertTrue(chunk.markers[0].valid)
        self.assertEqual(chunk.markers[0].position, (1.0, 2.0, 3.0))
        self.assertEqual(chunk.markers[0].id, marker_id)
        # Create and enable are one entry: a single undo removes both.
        doc.undo()
        self.assertEqual(chunk.markers, [])
        doc.redo()
        self.assertEqual(chunk.markers[0].label, "placed")

        # Moving an existing marker also makes it valid, and undoes in one step.
        chunk.markers[0].valid = False
        doc.undo()  # undo the explicit flag flip
        chunk.set_marker_position("placed", 4.0, 5.0, 6.0)
        self.assertEqual(chunk.markers[0].position, (4.0, 5.0, 6.0))
        self.assertTrue(chunk.markers[0].valid)

    def test_set_marker_label_validation(self):
        doc, chunk = empty_document()
        first = chunk.addMarker(label="one", position=(0.0, 0.0, 0.0))
        chunk.addMarker(label="two", position=(0.0, 0.0, 0.0))
        with self.assertRaises(Exception):
            first.label = ""
        with self.assertRaises(Exception):
            first.label = "two"
        self.assertEqual(chunk.markers[0].label, "one")


class TestTaskHistory(unittest.TestCase):
    def test_task_discards_undo_history(self):
        doc, chunk = empty_document()
        doc.active_chunk = chunk
        chunk.addMarker(label="before", position=(0.0, 0.0, 0.0))
        self.assertTrue(doc.can_undo)
        # A task mutates in place and cannot be undone; the history before it
        # no longer describes the document, so it is discarded.
        doc.duplicate_chunk(chunk)
        self.assertEqual(len(doc.chunk_ids), 2)
        self.assertFalse(doc.can_undo)
        self.assertFalse(doc.can_redo)


def editable_document():
    doc, chunk = empty_document()
    doc.active_chunk = chunk
    chunk.set_meta("note", "first")
    chunk.addMarker(label="GCP1", position=(1.0, 2.0, 3.0))
    return doc, chunk


class TestPersistence(unittest.TestCase):
    def test_save_directory_then_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc, _ = editable_document()
            doc.save(path)
            self.assertTrue(os.path.isfile(path))
            loaded = alphashape.load_project(path)
            self.assertEqual(len(loaded.chunk_ids), 1)
            chunk = loaded.chunk(loaded.chunk_ids[0])
            self.assertEqual(chunk.meta.get("note"), "first")
            self.assertEqual(chunk.markers[0].label, "GCP1")
            # path=None saves back to the document's current path.
            chunk.set_meta("note", "second")
            loaded.save()
            again = alphashape.load_project(path)
            self.assertEqual(
                again.chunk(again.chunk_ids[0]).meta.get("note"), "second")

    def test_save_psz_then_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psz")
            doc, _ = editable_document()
            doc.save(path)
            self.assertTrue(os.path.isfile(path))
            loaded = alphashape.load_project(path)
            self.assertEqual(len(loaded.chunk_ids), 1)
            self.assertEqual(loaded.chunk(loaded.chunk_ids[0]).markers[0].label,
                             "GCP1")
            # archive=True forces the single-file form even from a .psx path.
            doc.save(os.path.join(tmp, "forced.psx"), archive=True)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "forced.psz")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "forced.files")))

    def test_append_project(self):
        source, _ = editable_document()
        dest, dest_chunk = empty_document()
        # The AppendProject task resolves its target through the active chunk.
        dest.active_chunk = dest_chunk
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "source.psx")
            source.save(path)
            dest.append_project(path)
            self.assertEqual(len(dest.chunk_ids), 2)
            # Document.append takes an in-memory Document.
            other, _ = editable_document()
            dest.append(other)
            self.assertEqual(len(dest.chunk_ids), 3)

    def test_append_self_is_safe_and_replays(self):
        doc, chunk = editable_document()
        chunk.set_meta("note", "self")
        original_labels = [chunk.label for chunk in doc.chunks.values()]
        doc.append(doc)
        self.assertEqual(len(doc.chunk_ids), 2)
        doubled = [chunk.label for chunk in doc.chunks.values()]
        self.assertEqual(doubled, original_labels + original_labels)
        doc.undo()
        self.assertEqual(len(doc.chunk_ids), 1)
        doc.redo()
        self.assertEqual([chunk.label for chunk in doc.chunks.values()], doubled)

    def test_chunk_enabled_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "enabled.psx")
            doc, chunk = empty_document()
            chunk.enabled = False
            doc.save(path)
            loaded = alphashape.load_project(path)
            self.assertFalse(loaded.chunk(loaded.chunk_ids[0]).enabled)
            loaded.chunk(loaded.chunk_ids[0]).enabled = True
            loaded.save()
            again = alphashape.load_project(path)
            self.assertTrue(again.chunk(again.chunk_ids[0]).enabled)


class TestChunksApi(unittest.TestCase):
    def test_chunks_api(self):
        doc = alphashape.Document()
        first = doc.addChunk()
        self.assertEqual(first.label, "Chunk 1")
        second = doc.addChunk("Second")
        self.assertEqual([chunk.label for chunk in doc.chunks.values()],
                         ["Chunk 1", "Second"])
        self.assertEqual(len(doc.chunk_ids), 2)
        # addChunk leaves the active chunk alone.
        self.assertIsNone(doc.active_chunk)

        copy = first.copy("copy")
        self.assertEqual(copy.label, "copy")
        self.assertEqual(len(doc.chunk_ids), 3)

        doc.remove(second)
        self.assertEqual([chunk.label for chunk in doc.chunks.values()],
                         ["Chunk 1", "copy"])
        doc.undo()
        self.assertEqual(len(doc.chunk_ids), 3)
        doc.redo()
        self.assertEqual(len(doc.chunk_ids), 2)

        merged = doc.merge_chunks([first.id, copy.id], label="merged")
        self.assertEqual(merged.label, "merged")
        self.assertEqual(len(doc.chunk_ids), 3)
        self.assertEqual(doc.chunks[first.key].label, "Chunk 1")

    def test_add_chunk_default_labels(self):
        doc = alphashape.Document()
        self.assertEqual(doc.addChunk().label, "Chunk 1")
        self.assertEqual(doc.addChunk().label, "Chunk 2")
        # The next default skips names already in use.
        self.assertEqual(doc.addChunk().label, "Chunk 3")

    def test_add_chunk_redo_keeps_active(self):
        doc, first = empty_document()
        doc.active_chunk = first
        second = doc.addChunk()
        self.assertEqual(second.label, "Chunk 2")
        self.assertEqual(doc.active_chunk.id, first.id)
        doc.undo()
        doc.redo()
        self.assertEqual(doc.active_chunk.id, first.id)

    def test_chunk_arguments_accept_dict(self):
        doc, _ = empty_document()
        doc.addChunk("Second")
        merged = doc.merge_chunks(doc.chunks, label="merged")
        self.assertEqual(merged.label, "merged")
        # MergeChunks keeps the sources and appends the merged chunk.
        self.assertEqual(len(doc.chunk_ids), 3)
        # A dict argument contributes its VALUES, not its keys, and a document
        # must keep one chunk, so remove two of the three.
        subset = dict(list(doc.chunks.items())[:2])
        doc.remove(subset)
        self.assertEqual(len(doc.chunk_ids), 1)
        doc.undo()
        self.assertEqual(len(doc.chunk_ids), 3)


if __name__ == "__main__":
    unittest.main()
