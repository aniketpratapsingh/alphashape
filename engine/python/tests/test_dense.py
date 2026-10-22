#!/usr/bin/env python3
"""Dense-product views: Chunk.depth_maps and Chunk.point_cloud.

Milestone 6 added the BuildDepthMaps and BuildPointCloud tasks. They are reached
through the generic run_task surface exactly like DetectFeatures / MatchPhotos;
what this file pins is the read-only views onto their results:

  * depth_maps -> one dict per map, with camera_id/width/height/downscale and the
    HxW depth grid plus optional normals (HxWx3) and confidence (HxW).
  * point_cloud -> points (Nx3), optional normals/colors/confidence, point_count.

The views are plain nested lists (the module does not depend on numpy), matching
how Chunk.tracks and Camera.pose are shaped. The test also pins the two failure
modes the task exposes through the binder: an invalid downscale or filter_mode is
a ValueError, and an unknown keyword is an InvalidArgumentError.

Run with plain unittest, no pytest.
"""

import os
import unittest

import alphashape


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
# The rendered three-view scene used by the C++ matching tests and by the parity
# test's dense fixtures.
SCENE = [os.path.join(DATA_DIR, name) for name in ("scene_0.jpg", "scene_1.jpg", "scene_2.jpg")]

# 480x360 fixtures; downscale 4 keeps BuildDepthMaps fast (120x90 maps).
DOWNSCALE = 4


def dense_document(downscale=DOWNSCALE, **depth_kwargs):
    """A one-chunk document with a depth map per camera, but no fused cloud yet."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(SCENE))
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
    doc.run_task("AlignCameras")
    doc.run_task("BuildDepthMaps", downscale=downscale, **depth_kwargs)
    return doc, chunk


def fused_document(downscale=DOWNSCALE):
    """A dense_document with BuildPointCloud run on top of its depth maps."""
    doc, chunk = dense_document(downscale=downscale)
    doc.run_task("BuildPointCloud")
    return doc, chunk


class TestEmptyViews(unittest.TestCase):
    """Before the dense stage runs the containers are empty, not absent."""

    def test_fresh_chunk_has_no_dense_products(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))

        self.assertEqual(chunk.depth_maps, [])
        cloud = chunk.point_cloud
        self.assertEqual(cloud["point_count"], 0)
        self.assertEqual(cloud["points"], [])
        self.assertIsNone(cloud["normals"])
        self.assertIsNone(cloud["colors"])
        self.assertIsNone(cloud["confidence"])


class TestDepthMapView(unittest.TestCase):
    """Chunk.depth_maps exposes one map per reference camera, correctly shaped."""

    def test_one_map_per_camera_with_matching_shapes(self):
        _, chunk = dense_document()

        maps = chunk.depth_maps
        self.assertEqual(len(maps), len(chunk.cameras))

        for entry in maps:
            for key in ("camera_id", "width", "height", "downscale", "depth", "normals",
                        "confidence"):
                self.assertIn(key, entry)

            self.assertEqual(entry["downscale"], DOWNSCALE)
            # The fixtures are 480x360; a downscale of 4 gives 120x90.
            self.assertEqual(entry["width"], 120)
            self.assertEqual(entry["height"], 90)

            depth = entry["depth"]
            self.assertEqual(len(depth), entry["height"])
            for row in depth:
                self.assertEqual(len(row), entry["width"])
                for value in row:
                    self.assertIsInstance(value, float)
                    # -1 is the "no depth" sentinel; a real depth is positive.
                    self.assertTrue(value == -1.0 or value > 0.0)

            if entry["normals"] is not None:
                self.assertEqual(len(entry["normals"]), entry["height"])
                for row in entry["normals"]:
                    self.assertEqual(len(row), entry["width"])
                    for normal in row:
                        self.assertEqual(len(normal), 3)

            if entry["confidence"] is not None:
                self.assertEqual(len(entry["confidence"]), entry["height"])
                for row in entry["confidence"]:
                    self.assertEqual(len(row), entry["width"])

    def test_at_least_one_valid_depth(self):
        # A view that only ever produced sentinels would still pass the shaping
        # assertions above, so pin that the stage actually estimated depth.
        _, chunk = dense_document()
        valid = 0
        for entry in chunk.depth_maps:
            for row in entry["depth"]:
                for value in row:
                    if value != -1.0:
                        valid += 1
        self.assertGreater(valid, 0)

    def test_depth_downscale_is_recorded_in_meta(self):
        _, chunk = dense_document()
        self.assertEqual(chunk.meta["BuildDepthMaps/downscale"], str(DOWNSCALE))


class TestPointCloudView(unittest.TestCase):
    """Chunk.point_cloud exposes the fused cloud with the declared shapes."""

    def test_cloud_shapes_and_count(self):
        _, chunk = fused_document()
        cloud = chunk.point_cloud

        points = cloud["points"]
        self.assertEqual(cloud["point_count"], len(points))
        self.assertGreater(cloud["point_count"], 1000)

        for point in points:
            self.assertEqual(len(point), 3)
            for value in point:
                self.assertIsInstance(value, float)

        colors = cloud["colors"]
        self.assertIsNotNone(colors)
        self.assertEqual(len(colors), len(points))
        for color in colors:
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

        if cloud["normals"] is not None:
            self.assertEqual(len(cloud["normals"]), len(points))
        if cloud["confidence"] is not None:
            self.assertEqual(len(cloud["confidence"]), len(points))

    def test_cloud_outlives_document_deletion(self):
        doc, chunk = fused_document()
        count = chunk.point_cloud["point_count"]
        self.assertGreater(count, 0)
        del doc
        # The view is rebuilt per access from storage the chunk co-owns.
        self.assertEqual(chunk.point_cloud["point_count"], count)

    def test_cloud_survives_save_and_load(self):
        import tempfile

        doc, chunk = fused_document()
        before = chunk.point_cloud["point_count"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual(loaded.point_cloud["point_count"], before)


class TestDashboardArgumentValidation(unittest.TestCase):
    """The two dense tasks validate through the same binder as every other task."""

    def test_downscale_must_be_a_power_of_two(self):
        doc, chunk = dense_document()
        # 3 is not a power of two; the task raises the engine ValueError, which
        # the translator maps onto alphashape.ValueError.
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("BuildDepthMaps", downscale=3)

    def test_unknown_filter_mode_is_a_value_error(self):
        doc, chunk = dense_document()
        with self.assertRaises(alphashape.ValueError):
            doc.run_task("BuildDepthMaps", filter_mode="bogus")

    def test_unknown_keyword_is_an_invalid_argument(self):
        doc, chunk = dense_document()
        with self.assertRaises(alphashape.InvalidArgumentError) as ctx:
            doc.run_task("BuildDepthMaps", no_such_key=1)
        self.assertIn("no_such_key", str(ctx.exception))

    def test_point_cloud_rejects_unknown_keyword(self):
        doc, chunk = dense_document()
        with self.assertRaises(alphashape.InvalidArgumentError):
            doc.run_task("BuildPointCloud", no_such_key=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
