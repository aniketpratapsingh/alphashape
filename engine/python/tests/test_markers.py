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
AERIAL = [
    os.path.join(DATA_DIR, "aerial_r{}c{}.jpg".format(row, col))
    for row in range(3)
    for col in range(4)
]


def document_with_photos():
    """A document whose active chunk has two cameras with labels."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=[FIXTURE_A, FIXTURE_B])
    return doc, chunk


def write_disc_pgm(path, width=240, height=180, cx=120.0, cy=90.0, radius=28.0):
    """A white sheet with one dark filled disc, written as a binary PGM.

    The C++ renderer is not available from Python, but the image decoder reads
    PGM and the non-coded circular detector only needs a dark blob.
    """
    lines = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx = float(x) + 0.5 - cx
            dy = float(y) + 0.5 - cy
            row.append(30 if dx * dx + dy * dy <= radius * radius else 230)
        lines.append(bytes(row))
    with open(path, "wb") as handle:
        handle.write(b"P5\n%d %d\n255\n" % (width, height))
        handle.writelines(lines)


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


class TestDetectMarkers(unittest.TestCase):
    def test_detects_non_coded_circular(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "disc.pgm")
            write_disc_pgm(path)
            doc = alphashape.Document()
            chunk = doc.add_chunk("Chunk 1")
            doc.active_chunk = chunk
            doc.run_task("AddPhotos", filenames=[path])
            camera = chunk.cameras[0].label
            # AddPhotos leaves the camera unaligned; naming it explicitly is
            # what selects it (the default is "all enabled cameras").
            chunk.detectMarkers(target_type="Circular", cameras=[camera])
            self.assertEqual(len(chunk.markers), 1)
            self.assertEqual(chunk.markers[0]["label"], "point 1")
            self.assertEqual(chunk.meta["DetectMarkers/target_type"], "Circular")

    def test_unknown_target_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "disc.pgm")
            write_disc_pgm(path)
            doc = alphashape.Document()
            chunk = doc.add_chunk("Chunk 1")
            doc.active_chunk = chunk
            doc.run_task("AddPhotos", filenames=[path])
            with self.assertRaises(alphashape.ValueError) as context:
                chunk.detectMarkers(target_type="NotATarget",
                                    cameras=[chunk.cameras[0].label])
            message = str(context.exception)
            self.assertIn("target_type", message)
            for valid in ("Circular12", "Circular", "Cross", "AprilTag36h11"):
                self.assertIn(valid, message)

    def test_circle21_reports_the_documented_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "disc.pgm")
            write_disc_pgm(path)
            doc = alphashape.Document()
            chunk = doc.add_chunk("Chunk 1")
            doc.active_chunk = chunk
            doc.run_task("AddPhotos", filenames=[path])
            with self.assertRaises(alphashape.ValueError) as context:
                chunk.detectMarkers(target_type="AprilTagCircle21h7",
                                    cameras=[chunk.cameras[0].label])
            message = str(context.exception)
            self.assertIn("AprilTagCircle21h7", message)
            self.assertIn("not supported yet", message)
            self.assertIn("LIMITATIONS.md", message)


class TestRefineMarkers(unittest.TestCase):
    """The RefineMarkers task is bound and its parameters reach the project."""

    def test_binds_and_records_parameters(self):
        doc, chunk = document_with_photos()
        chunk.refineMarkers(patch_size=15, search_steps=4)
        self.assertEqual(chunk.meta["RefineMarkers/patch_size"], "15")
        self.assertEqual(chunk.meta["RefineMarkers/search_steps"], "4")

    def test_unknown_marker_label_is_rejected(self):
        doc, chunk = document_with_photos()
        with self.assertRaises(alphashape.ValueError) as context:
            chunk.refineMarkers(markers=["NO_SUCH_MARKER"])
        self.assertIn("NO_SUCH_MARKER", str(context.exception))


class TestTrackMarkers(unittest.TestCase):
    """The TrackMarkers task is bound and its parameters reach the project."""

    def test_binds_and_records_parameters(self):
        doc, chunk = document_with_photos()
        chunk.trackMarkers(window=15, direction="backward")
        self.assertEqual(chunk.meta["TrackMarkers/window"], "15")
        self.assertEqual(chunk.meta["TrackMarkers/direction"], "backward")

    def test_unknown_direction_is_rejected(self):
        doc, chunk = document_with_photos()
        with self.assertRaises(alphashape.ValueError) as context:
            chunk.trackMarkers(direction="sideways")
        self.assertIn("direction", str(context.exception))


class TestMarkerSurfaceExtensions(unittest.TestCase):
    """The WP-13.20 additions to the marker surface."""

    def test_add_marker_generates_a_label(self):
        _, chunk = document_with_photos()
        marker = chunk.addMarker()
        self.assertEqual(marker.label, "point 1")
        self.assertEqual(marker.type, "regular")

    def test_projection_is_a_marker_projection_view(self):
        _, chunk = document_with_photos()
        chunk.addMarker(label="GCP1")
        camera = chunk.cameras[0]
        chunk.setMarkerProjection("GCP1", camera.label, 10.0, 20.0)
        projection = chunk.markers[0].projections[0]
        self.assertEqual(projection.camera.label, camera.label)
        self.assertEqual(projection.camera_id, camera.id)
        self.assertEqual(projection.pixel, (10.0, 20.0))
        self.assertTrue(projection.enabled)
        # The earlier dict protocol still works.
        self.assertEqual(projection["camera_id"], camera.id)
        as_dict = dict(projection)
        self.assertEqual(set(as_dict),
                         {"camera_id", "camera", "pixel", "enabled"})
        self.assertEqual(as_dict["pixel"], (10.0, 20.0))
        self.assertEqual(len(projection), 4)
        self.assertEqual(list(projection), projection.keys())
        self.assertIsNone(projection.get("missing"))
        self.assertEqual(projection.get("missing", 7), 7)
        self.assertEqual(projection.get("pixel"), (10.0, 20.0))
        self.assertEqual([key for key, _ in projection.items()],
                         projection.keys())
        # Camera proxies compare by identity, not equality, so compare the
        # scalar values and the camera's id rather than the proxy objects.
        values = projection.values()
        self.assertEqual(len(values), 4)
        self.assertEqual(values[0], camera.id)
        self.assertEqual(values[1].id, camera.id)
        self.assertEqual(values[2], (10.0, 20.0))
        self.assertEqual(values[3], True)
        self.assertEqual([value for _, value in projection.items()][0], camera.id)

    def test_marker_group_round_trip(self):
        _, chunk = document_with_photos()
        group = chunk.addMarkerGroup("Control")
        marker = chunk.addMarker(label="GCP1")
        self.assertIsNone(marker.group)
        marker.group = group.id
        self.assertEqual(chunk.markers[0].group, group.id)
        marker.group = None
        self.assertIsNone(chunk.markers[0].group)

    def test_ground_control_separates_referenced_markers(self):
        _, chunk = document_with_photos()
        referenced = chunk.addMarker(label="GCP1")
        referenced.reference = (1.0, 2.0, 3.0)
        chunk.addMarker(label="NO_REFERENCE")
        control = chunk.ground_control
        self.assertEqual([m.label for m in control["markers"]], ["GCP1"])
        self.assertEqual([m.label for m in control["control_points"]], ["GCP1"])
        self.assertEqual(control["check_points"], [])

    def test_scalebar_round_trip(self):
        _, chunk = document_with_photos()
        first = chunk.addMarker(label="A")
        second = chunk.addMarker(label="B")
        bar = chunk.addScalebar(first.id, second.id, reference=1.5,
                                accuracy=0.001)
        self.assertEqual(bar.reference, 1.5)
        self.assertEqual(bar.accuracy, 0.001)
        self.assertEqual([b.id for b in chunk.scalebars], [bar.id])


class TestOptimizeCameras(unittest.TestCase):
    def test_requires_an_aligned_block(self):
        _, chunk = document_with_photos()
        self.assertTrue(chunk.use_gcps)
        with self.assertRaises(alphashape.ValueError) as context:
            chunk.optimizeCameras(use_gcps=False, adaptive_fitting=True)
        self.assertIn("align", str(context.exception))
        # The per-call override is restored even when the task fails.
        self.assertTrue(chunk.use_gcps)


class TestOptimizeCamerasUseGcps(unittest.TestCase):
    """`use_gcps` overrides the chunk setting for one call only."""

    @classmethod
    def setUpClass(cls):
        cls.doc = alphashape.Document()
        cls.chunk = cls.doc.add_chunk("Chunk 1")
        cls.doc.active_chunk = cls.chunk
        cls.doc.run_task("AddPhotos", filenames=list(AERIAL))
        cls.doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        cls.doc.run_task("AlignCameras")

    def test_per_call_override_leaves_the_chunk_setting_alone(self):
        self.assertTrue(self.chunk.use_gcps)
        self.chunk.optimizeCameras(use_gcps=False)
        self.assertTrue(self.chunk.use_gcps)
        # Default None uses the chunk's own value.
        self.chunk.optimizeCameras()
        self.assertTrue(self.chunk.use_gcps)

    def test_chunk_setting_is_persistent_and_undoable(self):
        self.chunk.use_gcps = False
        self.assertFalse(self.chunk.use_gcps)
        self.chunk.optimizeCameras()
        self.assertFalse(self.chunk.use_gcps)
        self.chunk.use_gcps = True
        self.assertTrue(self.chunk.use_gcps)


if __name__ == "__main__":
    unittest.main()
