#!/usr/bin/env python3
"""Calibration surface through the Python front end (WP-13.15/13.16/13.20).

What this file pins:

  * CalibrationFormat keeps its four existing values;
  * Sensor.calibration and Sensor.fixed_params expose the per-sensor model, and
    Calibration exposes p3, p4 and the (absent by default) covariance matrix;
  * Calibration.save/load round trip the model through the JSON codec;
  * Sensor.makeMaster detaches a slave sensor through the undoable command;
  * Chunk.set_sensor_fixed_params sets and releases the coefficients held fixed;
  * Chunk.calibrateCamera is bound and refuses a block with no detected board;
  * Chunk.exportSTMap writes a map for a calibrated sensor;
  * Chunk.detectFiducials accepts the public `fiducial_type` spelling.

No calibration board is processed, so the file runs in a couple of seconds.
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
CAM_A = os.path.join(DATA_DIR, "cam_a1.jpg")
CAM_B = os.path.join(DATA_DIR, "cam_b1.jpg")


def chunk_with(*paths):
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(paths))
    return doc, chunk


class TestCalibrationValues(unittest.TestCase):
    def test_enum_keeps_the_existing_values(self):
        # Four formats are kept; the neutral spellings are the documented ones.
        members = alphashape.CalibrationFormat.__members__
        self.assertEqual(len(members), 4)
        for name in ("Xml", "Json"):
            self.assertIn(name, members)

    def test_sensor_calibration_and_fixed_params(self):
        _, chunk = chunk_with(CAM_A)
        sensor = chunk.sensors[0]
        calibration = sensor.calibration
        self.assertGreater(calibration.f, 0.0)
        self.assertGreater(calibration.width, 0)
        self.assertEqual(calibration.p3, 0.0)
        self.assertEqual(calibration.p4, 0.0)
        self.assertIsNone(calibration.covariance_matrix)
        self.assertEqual(list(sensor.fixed_params), [])

    def test_save_and_load_json_round_trip(self):
        _, chunk = chunk_with(CAM_A)
        calibration = chunk.sensors[0].calibration
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibration.json")
            warnings = calibration.save(
                path, alphashape.CalibrationFormat.Json)
            self.assertEqual(list(warnings), [])
            loaded = alphashape.Calibration.load(
                path, alphashape.CalibrationFormat.Json)
        self.assertAlmostEqual(loaded.f, calibration.f, places=12)
        self.assertEqual(loaded.p3, calibration.p3)
        self.assertEqual(loaded.p4, calibration.p4)
        self.assertEqual(loaded.width, calibration.width)


class TestSensorMaster(unittest.TestCase):
    def test_make_master_detaches_a_slave(self):
        _, chunk = chunk_with(CAM_A, CAM_B)
        self.assertEqual(len(chunk.sensors), 2)
        chunk.set_sensor_master(1, 0)
        self.assertEqual(chunk.sensors[1].master, 0)
        chunk.sensors[1].makeMaster()
        self.assertEqual(chunk.sensors[1].master, -1)

    def test_make_master_on_an_independent_sensor_is_a_no_op(self):
        _, chunk = chunk_with(CAM_A)
        chunk.sensors[0].makeMaster()
        self.assertEqual(chunk.sensors[0].master, -1)


class TestSensorFixedParams(unittest.TestCase):
    """The fixed-coefficient list OptimizeCameras intersects its fit_* flags with.

    The command itself is pinned in test_model_commands.cpp; this file pins the
    Python route the GCP protocol's "fix the calibration from a controlled fit"
    variant uses, including the error contract.
    """

    def test_set_and_release_fixed_params(self):
        _, chunk = chunk_with(CAM_A)
        chunk.set_sensor_fixed_params(0, ["f", "k1"])
        self.assertEqual(list(chunk.sensors[0].fixed_params), ["f", "k1"])
        chunk.set_sensor_fixed_params(0, [])
        self.assertEqual(list(chunk.sensors[0].fixed_params), [])

    def test_unknown_and_duplicate_names_are_refused(self):
        _, chunk = chunk_with(CAM_A)
        with self.assertRaises(alphashape.ValueError):
            chunk.set_sensor_fixed_params(0, ["f", "not_a_coefficient"])
        with self.assertRaises(alphashape.ValueError):
            chunk.set_sensor_fixed_params(0, ["k2", "k2"])
        # A refused command leaves the chunk untouched.
        self.assertEqual(list(chunk.sensors[0].fixed_params), [])


class TestCalibrationTasks(unittest.TestCase):
    def test_calibrate_camera_refuses_a_block_without_a_board(self):
        _, chunk = chunk_with(CAM_A, CAM_B)
        with self.assertRaises(alphashape.Error) as context:
            chunk.calibrateCamera(board_cols=7, board_rows=5)
        self.assertIn("board", str(context.exception).lower())

    def test_export_stmap_writes_a_map(self):
        _, chunk = chunk_with(CAM_A)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "map.tif")
            chunk.exportSTMap(path=path, sensor=0, width=24, height=18)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_detect_fiducials_accepts_the_public_spelling(self):
        _, chunk = chunk_with(CAM_A)
        # No film sensor is configured, so the task visits no camera. The call
        # not raising also proves `fiducial_type` reached the task (an
        # unhandled keyword would be an InvalidArgumentError).
        chunk.detectFiducials(fiducial_type="circle")
        chunk.detectFiducials(detector="corner")

    def test_detect_fiducials_refuses_both_spellings(self):
        _, chunk = chunk_with(CAM_A)
        with self.assertRaises(alphashape.ValueError) as context:
            chunk.detectFiducials(fiducial_type="circle", detector="corner")
        self.assertIn("not both", str(context.exception))


if __name__ == "__main__":
    unittest.main()
