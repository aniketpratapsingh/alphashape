"""Tests for the MatchPhotos memory profile helpers (no CLI needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import match_profile  # noqa: E402


class ProgressTest(unittest.TestCase):
    def test_a_progress_line_names_its_stage(self):
        self.assertEqual(match_profile.parse_progress("[detect] 40%\n"), "detect")
        self.assertEqual(match_profile.parse_progress("[match] 100%"), "match")

    def test_other_lines_are_not_progress(self):
        for line in ("loading project", "[detect]", "[] 5%", "[detect] abc%", ""):
            self.assertIsNone(match_profile.parse_progress(line), line)


class TrackerTest(unittest.TestCase):
    def test_samples_go_to_the_current_stage_and_keep_the_peak(self):
        tracker = match_profile.StageTracker()
        tracker.enter("detect")
        tracker.sample(100)
        tracker.sample(300)
        tracker.sample(200)
        tracker.enter("after detect")
        tracker.sample(50)
        tracker.enter("match")
        tracker.sample(120)
        tracker.close()
        rows = {row["stage"]: row for row in match_profile._stages(tracker)}
        self.assertEqual(list(rows), ["detect", "after detect", "match"])
        self.assertEqual(tracker.peak["detect"], 300)
        self.assertEqual(match_profile._peak_mb(tracker), 0.0)  # 300 B rounds to 0 MB

    def test_re_entering_the_same_stage_is_one_stage(self):
        tracker = match_profile.StageTracker()
        tracker.enter("detect")
        tracker.enter("detect")
        tracker.close()
        self.assertEqual([row["stage"] for row in match_profile._stages(tracker)], ["detect"])


class SummaryTest(unittest.TestCase):
    def report(self, kp, threads, status="ok"):
        return {"dataset": "gcp-rtk", "images": 176, "megapixels": 20.0,
                "keypoint_limit_per_mpx": kp, "omp_num_threads": threads, "cpu_count": 4,
                "tasks": [{"task": "MatchPhotos", "status": status,
                           "stages": [{"stage": "detect", "peak_rss_mb": 5000.0, "seconds": 300.0}]}]}

    def test_rows_are_ordered_and_failures_are_shown(self):
        markdown = match_profile.summarize([self.report(1000, None, "failed: SIGKILL (out of memory?)"),
                                            self.report(250, "1")])
        rows = [line for line in markdown.splitlines() if line.startswith("| 2") or line.startswith("| 1")]
        self.assertTrue(rows[0].startswith("| 250 | 1 |"))
        self.assertIn("all (4)", markdown)
        self.assertIn("**failed: SIGKILL (out of memory?)**", markdown)
        self.assertIn("176 images at 20.0 MP", markdown)


if __name__ == "__main__":
    unittest.main()
