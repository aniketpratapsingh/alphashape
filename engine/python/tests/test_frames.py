#!/usr/bin/env python3
"""Camera grouping and frame statistics through Python (WP-11.8)."""

import os
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


class TestCameraGroups(unittest.TestCase):
    def test_station_grouping_then_clear(self):
        _, chunk = imported_document(CAM_A1, CAM_A2, CAM_B1)
        self.assertEqual(chunk.camera_groups, [])
        chunk.group_cameras("station")
        groups = chunk.camera_groups
        # The fixtures are three bodies with three distinct serials, so the
        # grouping is exactly one group per body, labelled by that serial.
        self.assertEqual(len(groups), 3)
        self.assertEqual(
            sorted(group["label"] for group in groups),
            ["AO100-0001", "AO100-0002", "BC7-0001"],
        )
        for group in groups:
            self.assertEqual(group["type"], "station")
        covered = sorted(cid for group in groups for cid in group["members"])
        self.assertEqual(covered, sorted(camera.id for camera in chunk.cameras))
        self.assertEqual(chunk.frame_stats()["group_count"], len(groups))

        chunk.group_cameras("none")
        self.assertEqual(chunk.camera_groups, [])

    def test_unknown_group_type_raises(self):
        _, chunk = imported_document(CAM_A1)
        with self.assertRaises(alphashape.ValueError):
            chunk.group_cameras("bogus")


class TestFrameStats(unittest.TestCase):
    def test_counts_follow_the_import(self):
        _, chunk = imported_document(CAM_A1, CAM_A2, CAM_B1)
        stats = chunk.frame_stats()
        self.assertEqual(stats["camera_count"], 3)
        self.assertEqual(stats["frame_count"], 3)
        self.assertEqual(stats["group_count"], 0)
        # AddPhotos leaves every camera unaligned.
        self.assertEqual(stats["enabled_count"], 0)
        self.assertEqual(stats["aligned_count"], 0)
        self.assertIsNone(stats["extent_min"])
        self.assertIsNone(stats["extent_max"])
        self.assertIsNone(stats["world_extent_min"])
        self.assertEqual(stats["mean_capture_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
