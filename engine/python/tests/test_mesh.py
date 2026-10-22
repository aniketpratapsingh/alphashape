#!/usr/bin/env python3
"""Mesh view + textures + export: Chunk.model, BuildUV/BuildTexture, ExportModel.

Milestone 7 added the BuildModel meshing task and the ExportModel /
ExportPointCloud PLY writers; milestone 8 added the BuildUV / BuildTexture
texturing tasks and the OBJ writer. What this file pins is:

  * Chunk.model -> {vertices (Nx3), faces (Fx3 int, in range), colors (Nx3
    uint8) / confidence / None, uvs (list of (u,v)) / None, uv_faces (list of
    index triples) / None, vertex_count, face_count}, shaped like the existing
    point_cloud view (plain nested lists, no numpy).
  * Chunk.model_texture -> None, or {width, height, channels, data: bytes}.
  * ExportModel writes a real PLY (magic + header) to a path, and an OBJ with a
    companion .mtl and .png when a texture page is present.
  * the failure modes the binder exposes: an unknown keyword is an
    InvalidArgumentError, a bad BuildModel source_data / BuildUV mapping_mode is
    a ValueError, and BuildTexture before BuildUV is a ValueError.

The pipeline runs on the three-view scene at downscale 4, matching the dense
test's fixture, so the whole file stays well under half a minute.

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
SCENE = [os.path.join(DATA_DIR, name) for name in ("scene_0.jpg", "scene_1.jpg", "scene_2.jpg")]

DOWNSCALE = 4
# A small volume (48^3) and a low face target keep BuildModel quick on this box.
VOLUME_RESOLUTION = 48


def meshed_document():
    """A one-chunk document with depth maps, a cloud and a small mesh."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(SCENE))
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
    doc.run_task("AlignCameras")
    doc.run_task("BuildDepthMaps", downscale=DOWNSCALE)
    doc.run_task("BuildPointCloud")
    doc.run_task("BuildModel", volume_resolution=VOLUME_RESOLUTION, face_count="low")
    return doc, chunk


def textured_document():
    """`meshed_document` plus a UV atlas and a small baked texture page."""
    doc, chunk = meshed_document()
    doc.run_task("BuildUV", texture_size=512)
    doc.run_task("BuildTexture", texture_size=512)
    return doc, chunk


class TestEmptyMeshView(unittest.TestCase):
    """Before BuildModel runs the mesh is empty, not absent."""

    def test_fresh_chunk_has_no_model(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))

        model = chunk.model
        for key in ("vertices", "faces", "colors", "confidence", "uvs", "uv_faces",
                    "vertex_count", "face_count"):
            self.assertIn(key, model)
        self.assertEqual(model["vertices"], [])
        self.assertEqual(model["faces"], [])
        self.assertIsNone(model["colors"])
        self.assertIsNone(model["confidence"])
        self.assertIsNone(model["uvs"])
        self.assertIsNone(model["uv_faces"])
        self.assertEqual(model["vertex_count"], 0)
        self.assertEqual(model["face_count"], 0)
        # No texture has been baked either.
        self.assertIsNone(chunk.model_texture)


class TestMeshView(unittest.TestCase):
    """Chunk.model exposes the extracted mesh with the declared shapes."""

    def test_mesh_shapes_and_indices(self):
        _, chunk = meshed_document()
        model = chunk.model

        vertices = model["vertices"]
        faces = model["faces"]
        self.assertGreater(model["vertex_count"], 0)
        self.assertGreater(model["face_count"], 0)
        self.assertEqual(model["vertex_count"], len(vertices))
        self.assertEqual(model["face_count"], len(faces))

        for vertex in vertices:
            self.assertEqual(len(vertex), 3)
            for value in vertex:
                self.assertIsInstance(value, float)

        for face in faces:
            self.assertEqual(len(face), 3)
            for index in face:
                self.assertIsInstance(index, int)
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(vertices))

    def test_mesh_has_colours(self):
        _, chunk = meshed_document()
        model = chunk.model
        colors = model["colors"]
        self.assertIsNotNone(colors)
        self.assertEqual(len(colors), len(model["vertices"]))
        for color in colors:
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_mesh_survives_save_and_load(self):
        doc, chunk = meshed_document()
        before = (chunk.model["vertex_count"], chunk.model["face_count"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual((loaded.model["vertex_count"], loaded.model["face_count"]), before)


class TestUvAndTexture(unittest.TestCase):
    """BuildUV lays out the mesh and BuildTexture bakes a page (milestone 8)."""

    def test_uv_layout_matches_face_count(self):
        _, chunk = textured_document()
        model = chunk.model
        uvs = model["uvs"]
        uv_faces = model["uv_faces"]
        self.assertIsNotNone(uvs)
        self.assertIsNotNone(uv_faces)
        # One index triple per face, in range of the coordinate pool.
        self.assertEqual(len(uv_faces), model["face_count"])
        self.assertGreater(len(uvs), 0)
        for uv in uvs:
            self.assertEqual(len(uv), 2)
            self.assertGreaterEqual(uv[0], 0.0)
            self.assertLessEqual(uv[0], 1.0)
            self.assertGreaterEqual(uv[1], 0.0)
            self.assertLessEqual(uv[1], 1.0)
        for triple in uv_faces:
            self.assertEqual(len(triple), 3)
            for index in triple:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(uvs))

    def test_texture_page_dimensions(self):
        _, chunk = textured_document()
        texture = chunk.model_texture
        self.assertIsNotNone(texture)
        self.assertEqual(texture["width"], 512)
        self.assertEqual(texture["height"], 512)
        self.assertEqual(texture["channels"], 3)
        self.assertIsInstance(texture["data"], bytes)
        self.assertEqual(len(texture["data"]), 512 * 512 * 3)

    def test_texture_survives_save_and_load(self):
        doc, chunk = textured_document()
        before = chunk.model_texture["data"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual(loaded.model_texture["data"], before)

    def test_build_texture_without_uv_is_a_value_error(self):
        doc, chunk = meshed_document()  # a model, but no BuildUV yet
        self.assertIsNone(chunk.model["uv_faces"])
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("BuildTexture", texture_size=256)

    def test_build_uv_unknown_mapping_is_a_value_error(self):
        doc, chunk = meshed_document()
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("BuildUV", mapping_mode="bogus")


class TestExport(unittest.TestCase):
    """ExportModel / ExportPointCloud through the binder write real PLY files."""

    def test_export_model_writes_a_ply(self):
        doc, chunk = meshed_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.ply")
            doc.run_task("ExportModel", path=path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(3), b"ply")

    def test_export_point_cloud_writes_a_ply(self):
        doc, chunk = meshed_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cloud.ply")
            doc.run_task("ExportPointCloud", path=path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(3), b"ply")

    def test_export_textured_model_writes_obj_mtl_and_png(self):
        doc, chunk = textured_document()
        with tempfile.TemporaryDirectory() as tmp:
            obj = os.path.join(tmp, "model.obj")
            doc.run_task("ExportModel", path=obj, format="obj")
            self.assertTrue(os.path.exists(obj))
            self.assertTrue(os.path.exists(os.path.join(tmp, "model.mtl")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "model.png")))
            with open(obj, "r") as handle:
                text = handle.read()
            self.assertIn("vt ", text)
            self.assertIn("mtllib model.mtl", text)


class TestArgumentValidation(unittest.TestCase):
    """The meshing and export tasks validate through the shared binder."""

    def test_build_model_unknown_source_is_a_value_error(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        doc.run_task("AlignCameras")
        doc.run_task("BuildDepthMaps", downscale=DOWNSCALE)
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("BuildModel", source_data="bogus")

    def test_export_empty_path_is_a_value_error(self):
        doc, chunk = meshed_document()
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("ExportModel", path="")

    def test_export_unknown_keyword_is_an_invalid_argument(self):
        doc, chunk = meshed_document()
        with self.assertRaises(alphashape.InvalidArgumentError):
            doc.run_task("ExportModel", no_such_key=1)

    def test_export_model_without_a_mesh_fails(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(alphashape.Error):
                doc.run_task("ExportModel", path=os.path.join(tmp, "none.ply"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
