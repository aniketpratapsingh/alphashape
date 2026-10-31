#!/usr/bin/env python3
"""Parity ("drift") test: the Python front end and the project file agree.

The whole architecture rests on one claim: a task declares its parameters once,
and every front end binds against that same schema. The strongest way to test
that claim is to compare two things that were produced by *different* code paths:

  * the parameter names reachable through the Python binder, and
  * the parameter keys the engine writes into a saved project.

If a task ever gains a parameter that the binder does not route (or the writer
records a key the binder would reject), the two sets diverge and this test fails.

Run with plain unittest, no pytest.
"""

import gc
import os
import sys
import tempfile
import unittest

import alphashape


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
FIXTURE_A = os.path.join(DATA_DIR, "fixture_a.pgm")
FIXTURE_B = os.path.join(DATA_DIR, "fixture_b.pgm")
CAM_A1 = os.path.join(DATA_DIR, "cam_a1.jpg")
CAM_A2 = os.path.join(DATA_DIR, "cam_a2.jpg")
CAM_B1 = os.path.join(DATA_DIR, "cam_b1.jpg")
# The rendered three-view scene used by the C++ matching tests.
SCENE = [os.path.join(DATA_DIR, name) for name in ("scene_0.jpg", "scene_1.jpg", "scene_2.jpg")]


def make_document_with_images(*paths):
    """A one-chunk document with each image imported through AddPhotos."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(paths))
    return doc, chunk


# The two drift-invariant tests run the SAME prepared pipeline for every task.
# Building it twice (once per test method) roughly doubles the class's runtime,
# because the dense + BuildModel + export pipelines dominate it. The prepared
# document is therefore built once per task and reused; neither test method
# mutates geometry, so sharing it cannot change any assertion.
_PREPARED_CACHE = {}


def _parameter_keys(meta, task):
    """The "<Task>/<param>" keys in `meta` that are PARAMETERS.

    A parameter key has exactly two segments. A task may also record other
    results under its own prefix with a deeper key -- AlignChunks writes the
    applied similarity to "AlignChunks/<chunk id>/matrix" (§4.8) -- and those
    are results, not declared parameters, so they are excluded here.
    """
    prefix = task + "/"
    return {
        key
        for key in meta
        if key.startswith(prefix) and key.count("/") == 1
    }


class TestDriftInvariant(unittest.TestCase):
    """Every parameter a task declares must reach the project meta, and back."""

    def _prepared_document(self, task):
        """A document with `task` run once, plus any prerequisites it needs.

        Most tasks run on the two generic fixtures. AlignCameras, OptimizeCameras
        and the dense tasks (BuildDepthMaps, BuildPointCloud, BuildModel) need tie
        points, which those fixtures do not produce, so they run on the matched
        three-view scene instead. BuildPointCloud additionally needs depth maps,
        and the milestone-9 tasks need a DEM (and, for the ortho tasks, an
        orthomosaic) on top of the dense products,
        and BuildModel needs either depth maps or a point cloud, so BuildDepthMaps
        (and for BuildModel a point cloud too) is run first. The export tasks need
        the geometry they write plus a destination path.

        The chunk-planning tasks (WP-11.6) are the exception: DuplicateChunk and
        MergeChunks only restructure chunks, AppendProject needs a second project
        on disk, and SaveChunksSubset is a Document method, not a task.
        """
        if task in _PREPARED_CACHE:
            doc, chunk = _PREPARED_CACHE[task]
            # Return a fresh chunk handle onto the cached document; the tests
            # only read it (and save/load it), never mutate the geometry.
            return doc, chunk
        if task == "AppendProject":
            # Needs a second project on disk to append from; the incoming
            # chunks arrive with offset ids, and run_task records the
            # parameters into the destination chunk that was active first.
            with tempfile.TemporaryDirectory() as tmp:
                source = alphashape.Document()
                source_chunk = source.add_chunk("Source")
                source.active_chunk = source_chunk
                source.run_task("AddPhotos", filenames=[CAM_A1, CAM_B1])
                path = os.path.join(tmp, "source.psx")
                source.save(path)

                doc, chunk = make_document_with_images(FIXTURE_A, FIXTURE_B)
                doc.run_task(task, path=path)
                # The task makes the appended chunk active; the drift checks
                # read the chunk run_task recorded its parameters into.
                doc.active_chunk = chunk
            _PREPARED_CACHE[task] = (doc, chunk)
            return doc, chunk
        if task == "AlignChunks":
            # Marker-based alignment (WP-11.7): two chunks of the same frames,
            # marker labels matching, both with estimated positions. The
            # parameters are recorded into the ACTIVE chunk, which must stay
            # active for the drift checks below.
            doc, chunk = make_document_with_images(FIXTURE_A, FIXTURE_B)
            second = doc.add_chunk("Reference")
            points = [(0.0, 0.0, 0.0), (10.0, 0.0, 2.0), (0.0, 8.0, 4.0),
                      (6.0, 6.0, 1.0)]
            for i, (x, y, z) in enumerate(points):
                label = "M{}".format(i + 1)
                chunk.set_marker_position(label, x, y, z)
                second.set_marker_position(label, x + 1.0, y - 2.0, z + 0.5)
            doc.active_chunk = chunk
            doc.run_task(task, reference_chunk_id=second.id)
            doc.active_chunk = chunk
            _PREPARED_CACHE[task] = (doc, chunk)
            return doc, chunk
        if task in ("DuplicateChunk", "MergeChunks"):
            # Chunk-planning tasks: they restructure chunks only, so the plain
            # imported document is all they need. MergeChunks needs two sources.
            doc, chunk = make_document_with_images(FIXTURE_A, FIXTURE_B)
            if task == "MergeChunks":
                doc.add_chunk("Second")
                doc.active_chunk = chunk
            doc.run_task(task)
            doc.active_chunk = chunk
            _PREPARED_CACHE[task] = (doc, chunk)
            return doc, chunk
        if task in ("AlignCameras", "OptimizeCameras", "BuildDepthMaps", "BuildPointCloud",
                    "BuildModel", "BuildUV", "BuildTexture", "ExportModel", "ExportPointCloud",
                    "BuildDem", "BuildOrthomosaic", "ExportDem", "ExportOrthomosaic"):
            doc, chunk = make_document_with_images(*SCENE)
            doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
            doc.run_task("AlignCameras")
            if task == "OptimizeCameras":
                doc.run_task("OptimizeCameras")
            elif task in ("BuildDepthMaps", "BuildPointCloud", "BuildModel", "BuildUV",
                          "BuildTexture", "ExportModel", "ExportPointCloud",
                          "BuildDem", "BuildOrthomosaic", "ExportDem", "ExportOrthomosaic"):
                # filter_mode="mild" is pinned: this file checks the parameter
                # contract (declared keys reach the meta and survive a save), not
                # the filter's strength. At the 120x90 working resolution of this
                # fixture the moderate default keeps well under half the pixels,
                # which starves the small BuildModel volume used below.
                doc.run_task("BuildDepthMaps", filter_mode="mild")
                if task in ("BuildPointCloud", "BuildModel", "BuildUV", "BuildTexture",
                            "ExportModel", "ExportPointCloud"):
                    doc.run_task("BuildPointCloud")
                if task in ("BuildModel", "BuildUV", "BuildTexture", "ExportModel"):
                    # Small volume + face target keep the parity run quick;
                    # ExportModel needs a mesh, ExportPointCloud only the cloud.
                    doc.run_task("BuildModel", volume_resolution=32, face_count="custom",
                                 face_count_custom=2000)
                if task in ("BuildUV", "BuildTexture"):
                    # BuildTexture needs a UV layout, so BuildUV runs first.
                    doc.run_task("BuildUV", texture_size=256)
                if task == "BuildTexture":
                    doc.run_task("BuildTexture", texture_size=256)
                # Milestone 9: all four product tasks need a DEM (the ortho
                # tasks use it as their surface too).
                if task in ("BuildDem", "BuildOrthomosaic", "ExportDem",
                            "ExportOrthomosaic"):
                    doc.run_task("BuildDem")
                if task in ("BuildOrthomosaic", "ExportOrthomosaic"):
                    doc.run_task("BuildOrthomosaic", refine_seamlines=False,
                                 color_correction="none")
                if task in ("ExportModel", "ExportPointCloud"):
                    # The export tasks need a real destination path.
                    with tempfile.TemporaryDirectory() as tmp:
                        doc.run_task(task, path=os.path.join(tmp, "export.ply"))
                if task in ("ExportDem", "ExportOrthomosaic"):
                    with tempfile.TemporaryDirectory() as tmp:
                        doc.run_task(task, path=os.path.join(tmp, "export.tif"))
            _PREPARED_CACHE[task] = (doc, chunk)
            return doc, chunk
        doc, chunk = make_document_with_images(FIXTURE_A, FIXTURE_B)
        if task != "AddPhotos":  # already run once by the helper
            doc.run_task(task)
        _PREPARED_CACHE[task] = (doc, chunk)
        return doc, chunk

    def test_declared_params_equal_recorded_meta_keys(self):
        for task in alphashape.list_tasks():
            params = alphashape.task_params(task)
            expected = {"{}/{}".format(task, name) for name in params}

            doc, chunk = self._prepared_document(task)
            meta = chunk.meta
            recorded = _parameter_keys(meta, task)

            self.assertEqual(
                recorded,
                expected,
                "task {}: recorded meta keys {} != declared params {}".format(
                    task, sorted(recorded), sorted(expected)
                ),
            )

    def test_recorded_keys_survive_save_and_load(self):
        for task in alphashape.list_tasks():
            expected = {"{}/{}".format(task, name) for name in alphashape.task_params(task)}

            doc, chunk = self._prepared_document(task)
            before = dict(chunk.meta)

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "project.psx")
                doc.save(path)
                reloaded = alphashape.load_project(path)

            loaded_chunk = reloaded.chunk(reloaded.active_chunk.id)
            self.assertIsNotNone(loaded_chunk)
            after = dict(loaded_chunk.meta)
            # The full meta must round-trip unchanged, not merely the task keys:
            # a save that dropped or reordered properties would be a real defect.
            self.assertEqual(before, after, "task {}: meta changed across save/load".format(task))
            recorded = _parameter_keys(after, task)
            self.assertEqual(recorded, expected)


class TestKeyboardBinder(unittest.TestCase):
    """The binder rejects what the schema does not declare, by name."""

    def test_unknown_kwarg_is_invalid_argument(self):
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(alphashape.InvalidArgumentError) as ctx:
            doc.run_task("DetectFeatures", no_such_param=5)
        self.assertIn("no_such_param", str(ctx.exception))

    def test_unknown_task_is_unsupported(self):
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(alphashape.UnsupportedTaskError):
            doc.run_task("NoSuchTask")

    def test_wrong_typed_kwarg_is_type_error(self):
        # Documented contract: a value whose Python type cannot satisfy the
        # declared type raises Python's TypeError (the module raises py::type_error).
        # It is NOT a ValueError: the request was well-formed, the value was not.
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(TypeError):
            doc.run_task("DetectFeatures", keypoint_limit="x")


class TestStringListBinder(unittest.TestCase):
    """A StringList parameter binds a list or tuple of str and nothing else."""

    def test_bare_string_is_rejected(self):
        # The classic bug this guards against: a str IS iterable, so accepting
        # it would silently import the characters of a single file name.
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(TypeError):
            doc.run_task("AddPhotos", filenames=CAM_A1)

    def test_non_string_element_is_rejected(self):
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(TypeError):
            doc.run_task("AddPhotos", filenames=[CAM_A1, 5])

    def test_non_sequence_is_rejected(self):
        doc, _ = make_document_with_images(FIXTURE_A)
        with self.assertRaises(TypeError):
            doc.run_task("AddPhotos", filenames=42)

    def test_tuple_is_accepted(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=(CAM_A1, CAM_B1))
        self.assertEqual(chunk.camera_count, 2)


class TestSensorAndCameraViews(unittest.TestCase):
    """Chunk.sensors and Chunk.cameras expose the imported geometry."""

    def test_sensors_and_cameras_reflect_the_import(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2, CAM_B1)

        sensors = chunk.sensors
        self.assertEqual(len(sensors), 2)
        by_model = {}
        for sensor in sensors:
            self.assertEqual(sensor.type, "frame")
            self.assertEqual(sensor.width, 200)
            self.assertEqual(sensor.height, 150)
            self.assertEqual(sensor.calibration.type, "frame")
            self.assertEqual(sensor.calibration.width, sensor.width)
            self.assertEqual(sensor.calibration.height, sensor.height)
            # Every route (focal-plane, 35mm, fallback) yields a positive focal
            # length; the exact value differs per fixture and is asserted in the
            # C++ tests where read_exif is available.
            self.assertGreater(sensor.calibration.f, 0.0)
            self.assertEqual(sensor.calibration.cx, 0.0)
            self.assertEqual(sensor.calibration.cy, 0.0)
            by_model[sensor.model] = sensor
        # AO-100 has a focal-plane resolution, so the pixel pitch is known; the
        # BC-7 fixture deliberately has none, so it stays 0 (unknown).
        self.assertGreater(by_model["AO-100"].pixel_size_um, 0.0)
        self.assertEqual(by_model["BC-7"].pixel_size_um, 0.0)

        cameras = chunk.cameras
        self.assertEqual(len(cameras), 3)
        a1 = [c for c in cameras if c.label == "cam_a1"][0]
        a2 = [c for c in cameras if c.label == "cam_a2"][0]
        b1 = [c for c in cameras if c.label == "cam_b1"][0]
        self.assertEqual(a1.sensor_id, a2.sensor_id)
        self.assertNotEqual(a1.sensor_id, b1.sensor_id)
        self.assertEqual(a1.orientation, 1)
        self.assertFalse(a1.enabled)
        self.assertIsNone(b1.reference)
        self.assertIsNotNone(a2.reference)
        lon, lat, alt = a2.reference
        self.assertLess(lon, 0.0)
        self.assertLess(lat, 0.0)

        # The per-axis accuracy view mirrors the reference: None where there is
        # no location, and a default 10 m isotropic sigma where there is one
        # (these fixtures carry no per-axis accuracy, so the default applies).
        self.assertIsNone(b1.reference_accuracy)  # cam_b1 has no GPS
        self.assertIsNotNone(a1.reference_accuracy)
        self.assertEqual(a1.reference_accuracy, (10.0, 10.0, 10.0))
        self.assertEqual(a2.reference_accuracy, (10.0, 10.0, 10.0))

    def test_views_do_not_dangle_when_the_chunk_grows(self):
        doc, chunk = make_document_with_images(CAM_A1)
        cameras_before = chunk.cameras
        # Importing more photos reallocates the chunk's camera vector; the view
        # grabbed earlier must still be valid, because it is a copy.
        doc.run_task("AddPhotos", filenames=[CAM_B1])
        self.assertEqual(cameras_before[0].label, "cam_a1")
        self.assertEqual(len(chunk.cameras), 2)


class TestSemantics(unittest.TestCase):
    """A bound value must actually take effect, and be recorded verbatim."""

    def test_keypoint_limit_caps_every_camera(self):
        # First confirm the fixtures are rich enough that the cap is the binding
        # constraint; otherwise "<= 30" could pass on an image with 3 keypoints.
        uncapped_doc, uncapped_chunk = make_document_with_images(FIXTURE_A)
        uncapped_doc.run_task("DetectFeatures", keypoint_limit=0, keypoint_limit_per_mpx=0)
        self.assertGreater(
            uncapped_chunk.keypoint_count,
            30,
            "fixture too sparse: the cap test would pass without the cap binding",
        )

        doc, chunk = make_document_with_images(FIXTURE_A, FIXTURE_B)
        doc.run_task("DetectFeatures", keypoint_limit=30)

        self.assertGreater(chunk.camera_count, 0)
        for camera in range(chunk.camera_count):
            count = len(chunk.keypoints(camera))
            self.assertLessEqual(count, 30, "camera {} exceeded the cap".format(camera))
            # The cap must actually bind on this fixture, not merely not-exceed it.
            self.assertEqual(count, 30, "camera {} was not truncated to the cap".format(camera))
        self.assertEqual(chunk.meta["DetectFeatures/keypoint_limit"], "30")


class TestMatchAndTrackViews(unittest.TestCase):
    """Chunk.match_pair_count / matches() / track_count / tracks().

    Until the MatchPhotos task lands, a freshly imported chunk carries no
    matches or tracks, so these tests pin the empty and zero results and the
    argument normalisation of matches(); they deliberately do not exercise a
    populated chunk (that belongs with MatchPhotos parity).
    """

    def test_empty_counts_and_views(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2, CAM_B1)
        self.assertEqual(chunk.match_pair_count, 0)
        self.assertEqual(chunk.track_count, 0)
        self.assertEqual(chunk.tracks(), [])

    def test_unknown_pair_returns_empty_list(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2)
        # No matches have been computed, so any pair is unknown.
        self.assertEqual(chunk.matches(0, 1), [])
        # An argument order that is not ascending must not be treated as
        # "unknown" merely because the ids are swapped: it normalises first.
        self.assertEqual(chunk.matches(1, 0), [])

    def test_argument_order_is_normalised(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2, CAM_B1)
        # Both orders and both extremes (camera ids 0..2) must agree, and both
        # must be empty on an un-matched chunk. A normalisation bug that only
        # looked for camera_a < camera_b in the stored order would still return
        # [] here, so we also assert the two calls agree with each other.
        for a in range(chunk.camera_count):
            for b in range(chunk.camera_count):
                self.assertEqual(chunk.matches(a, b), chunk.matches(b, a))

    def test_tracks_view_is_a_list(self):
        doc, chunk = make_document_with_images(CAM_A1)
        self.assertIsInstance(chunk.tracks(), list)

    def test_matches_survive_a_grab_and_reload(self):
        # The empty views must still be empty after a save/load round-trip, so a
        # reader that fabricated records would be caught.
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded_chunk = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual(loaded_chunk.match_pair_count, 0)
        self.assertEqual(loaded_chunk.track_count, 0)
        self.assertEqual(loaded_chunk.tracks(), [])
        self.assertEqual(loaded_chunk.matches(0, 1), [])


class TestMatchPhotosParity(unittest.TestCase):
    """MatchPhotos through the Python binder on the three-view scene."""

    def _matched_chunk(self, **kwargs):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000, **kwargs)
        return doc, chunk

    def test_three_pairs_and_tracks(self):
        _, chunk = self._matched_chunk()
        self.assertEqual(chunk.match_pair_count, 3)
        self.assertGreater(chunk.track_count, 0)

    def test_matches_argument_order_agrees(self):
        _, chunk = self._matched_chunk()
        for a in range(chunk.camera_count):
            for b in range(chunk.camera_count):
                # Both argument orders must agree, and the ascending pair must
                # be non-empty for the three real camera pairs.
                self.assertEqual(chunk.matches(a, b), chunk.matches(b, a))
        self.assertGreater(len(chunk.matches(1, 0)), 0)

    def test_drift_invariant_for_match_photos(self):
        # The keys reachable through the binder equal those recorded in meta,
        # before and after a save/load round-trip.
        expected = {"MatchPhotos/{}".format(name)
                    for name in alphashape.task_params("MatchPhotos")}
        doc, chunk = self._matched_chunk()
        recorded = {key for key in chunk.meta if key.startswith("MatchPhotos/")}
        self.assertEqual(recorded, expected)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        recorded_after = {key for key in loaded.meta if key.startswith("MatchPhotos/")}
        self.assertEqual(recorded_after, expected)

    def test_reset_matches_is_deterministic(self):
        _, chunk = self._matched_chunk()
        first = (chunk.match_pair_count, chunk.track_count, chunk.matches(0, 1))

        doc2 = alphashape.Document()
        chunk2 = doc2.add_chunk("Chunk 1")
        doc2.active_chunk = chunk2
        doc2.run_task("AddPhotos", filenames=list(SCENE))
        doc2.run_task("MatchPhotos", keypoint_limit_per_mpx=10000, reset_matches=True)
        second = (chunk2.match_pair_count, chunk2.track_count, chunk2.matches(0, 1))
        self.assertEqual(first, second)


class TestSfMViews(unittest.TestCase):
    """Chunk.aligned_camera_count / tie_point_count / tie_points(), Camera.pose
    and Camera.center, on a freshly imported chunk.

    Until AlignCameras lands and populates points, an imported chunk has no
    aligned cameras and no tie points, so these tests pin the empty and zero
    results and the identity-pose centre of an unaligned camera.
    """

    def test_empty_sfm_counts_and_points(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2, CAM_B1)
        self.assertEqual(chunk.aligned_camera_count, 0)
        self.assertEqual(chunk.tie_point_count, 0)
        self.assertEqual(chunk.tie_points(), [])

    def test_enabled_is_exposed_and_unaligned_by_default(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2)
        for camera in chunk.cameras:
            # AddPhotos imports every camera unaligned.
            self.assertFalse(camera.enabled)

    def test_pose_and_center_of_an_unaligned_camera(self):
        doc, chunk = make_document_with_images(CAM_A1)
        camera = chunk.cameras[0]
        # The pose is the identity 4x4, so the centre C = -R^T t is (0, 0, 0).
        pose = camera.pose
        self.assertEqual(len(pose), 4)
        for r in range(4):
            self.assertEqual(len(pose[r]), 4)
            for c in range(4):
                self.assertEqual(pose[r][c], 1.0 if r == c else 0.0)
        center = camera.center
        self.assertEqual(len(center), 3)
        self.assertEqual(center, (0.0, 0.0, 0.0))

    def test_sfm_views_survive_a_grab_and_reload(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual(loaded.aligned_camera_count, 0)
        self.assertEqual(loaded.tie_point_count, 0)
        self.assertEqual(loaded.tie_points(), [])
        self.assertEqual(loaded.cameras[0].center, (0.0, 0.0, 0.0))


class TestTransformAndCrsViews(unittest.TestCase):
    """Chunk.transform and Chunk.crs_epsg on a scene with no GNSS references.

    The three-view scene carries no GPS, so georeferencing cannot run and the
    chunk must stay in its arbitrary frame: the transform is the identity 4x4
    and the EPSG code is 0.
    """

    def test_identity_transform_and_unset_crs_on_non_gps_scene(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        doc.run_task("AlignCameras")

        transform = chunk.transform
        self.assertEqual(len(transform), 4)
        for r in range(4):
            self.assertEqual(len(transform[r]), 4)
            for c in range(4):
                self.assertEqual(transform[r][c], 1.0 if r == c else 0.0)
        self.assertEqual(chunk.crs_epsg, 0)

    def test_transform_and_crs_survive_a_grab_and_reload(self):
        doc, chunk = make_document_with_images(CAM_A1, CAM_A2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        self.assertEqual(loaded.crs_epsg, 0)
        self.assertEqual(loaded.transform[0], [1.0, 0.0, 0.0, 0.0])


class TestAlignCamerasParity(unittest.TestCase):
    """AlignCameras through the Python binder on the three-view scene."""

    def _aligned_chunk(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        doc.run_task("AlignCameras")
        return doc, chunk

    def test_alignment_produces_cameras_and_points(self):
        _, chunk = self._aligned_chunk()
        self.assertEqual(chunk.aligned_camera_count, 3)
        self.assertGreater(chunk.tie_point_count, 50)
        # A genuine reconstruction places the cameras apart, not all on top of
        # one another.
        centres = [camera.center for camera in chunk.cameras]
        self.assertGreater(len(centres), 1)
        self.assertNotEqual(centres[0], centres[1])

    def test_drift_invariant_for_align_cameras(self):
        # The keys reachable through the binder equal those recorded in meta,
        # before and after a save/load round-trip.
        expected = {"AlignCameras/{}".format(name)
                    for name in alphashape.task_params("AlignCameras")}
        doc, chunk = self._aligned_chunk()
        recorded = {key for key in chunk.meta if key.startswith("AlignCameras/")}
        self.assertEqual(recorded, expected)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        recorded_after = {key for key in loaded.meta if key.startswith("AlignCameras/")}
        self.assertEqual(recorded_after, expected)


class TestOptimizeCamerasParity(unittest.TestCase):
    """OptimizeCameras through the Python binder on the three-view scene."""

    def _optimized_chunk(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(SCENE))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        doc.run_task("AlignCameras")
        doc.run_task("OptimizeCameras")
        return doc, chunk

    def test_optimization_keeps_the_reconstruction(self):
        _, chunk = self._optimized_chunk()
        self.assertEqual(chunk.aligned_camera_count, 3)
        self.assertGreater(chunk.tie_point_count, 50)
        centres = [camera.center for camera in chunk.cameras]
        self.assertGreater(len(centres), 1)
        self.assertNotEqual(centres[0], centres[1])

    def test_drift_invariant_for_optimize_cameras(self):
        expected = {"OptimizeCameras/{}".format(name)
                    for name in alphashape.task_params("OptimizeCameras")}
        doc, chunk = self._optimized_chunk()
        recorded = {key for key in chunk.meta if key.startswith("OptimizeCameras/")}
        self.assertEqual(recorded, expected)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
        loaded = reloaded.chunk(reloaded.active_chunk.id)
        recorded_after = {key for key in loaded.meta if key.startswith("OptimizeCameras/")}
        self.assertEqual(recorded_after, expected)


class TestLifetime(unittest.TestCase):
    """A Chunk handed to Python must outlive the Document it came from."""

    def test_chunk_survives_document_deletion(self):
        doc, chunk = make_document_with_images(FIXTURE_A)
        chunk.label = "kept"
        del doc
        gc.collect()
        # Would be a use-after-free if Chunk did not co-own its storage.
        self.assertEqual(chunk.label, "kept")
        self.assertEqual(chunk.frame_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
