#!/usr/bin/env python3
"""The sample dataset generator (tools/sample/make_sample.py).

What this pins, without needing a built engine:

  * the generator runs and writes the 12 photographs, the manifest and the
    provenance files;
  * every image is a readable 480x360 JPEG carrying GPS EXIF, so the block
    georeferences itself the way the sample's README promises;
  * the manifest agrees with what was written (files exist, one entry per image,
    the licence is stated);
  * the output is deterministic: two runs are byte-identical, which is what
    makes the dataset safe to publish from CI.

Skipped (not failed) when numpy or Pillow is missing, so the file can be run on
a bare interpreter; CI installs both before it renders the dataset.

Run with plain unittest, no pytest.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "tools", "sample", "make_sample.py")

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401

    HAVE_DEPS = True
except ImportError:  # pragma: no cover - only on an interpreter without the deps
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are needed to render the sample")
class TestSampleGenerator(unittest.TestCase):
    """One render, shared by every check: the generator is the slow part."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls._tmp.name, "run1")
        result = subprocess.run(
            [sys.executable, SCRIPT, "--out", cls.out],
            cwd=REPO, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise AssertionError(
                "make_sample.py failed (%d):\n%s\n%s"
                % (result.returncode, result.stdout, result.stderr))
        cls.root = os.path.join(cls.out, "alphashape-sample")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_writes_the_expected_tree(self):
        for name in ("manifest.json", "README.md", "SOURCE.md"):
            path = os.path.join(self.root, name)
            self.assertTrue(os.path.isfile(path), name)
            self.assertGreater(os.path.getsize(path), 0, name)
        images = sorted(os.listdir(os.path.join(self.root, "images")))
        self.assertEqual(len(images), 12, images)
        self.assertEqual(images[0], "aerial_r0c0.jpg")
        self.assertEqual(images[-1], "aerial_r2c3.jpg")

    def test_images_are_gps_tagged_jpegs(self):
        from PIL import Image
        for name in ("aerial_r0c0.jpg", "aerial_r2c3.jpg"):
            path = os.path.join(self.root, "images", name)
            with Image.open(path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (480, 360))
                exif = image.getexif()
                gps = exif.get_ifd(0x8825)  # the GPS IFD
            self.assertTrue(gps, "no GPS IFD in " + name)
            self.assertEqual(gps[0x0001], "N")   # GPSLatitudeRef
            self.assertEqual(gps[0x0003], "E")   # GPSLongitudeRef
            self.assertIn(0x0002, gps)           # GPSLatitude
            self.assertIn(0x0004, gps)           # GPSLongitude
            self.assertIn(0x0006, gps)           # GPSAltitude

    def test_manifest_describes_what_was_written(self):
        with open(os.path.join(self.root, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["name"], "alphashape-sample")
        self.assertEqual(manifest["licence"], "CC0-1.0")
        self.assertEqual(len(manifest["images"]), 12)
        for entry in manifest["images"]:
            path = os.path.join(self.root, entry["file"])
            self.assertTrue(os.path.isfile(path), entry["file"])
            self.assertEqual(entry["width"], 480)
            self.assertEqual(entry["height"], 360)
            self.assertEqual(len(entry["R"]), 3)
            self.assertEqual(len(entry["R"][0]), 3)
            self.assertEqual(len(entry["t"]), 3)
            for key in ("lat", "lon", "height"):
                self.assertIn(key, entry["gps"])

    def test_the_block_spans_the_documented_area(self):
        # The scene is 120 m x 90 m; the camera centres must sit inside it
        # horizontally and 60 m up, which is what makes the overlap claim true.
        with open(os.path.join(self.root, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        for entry in manifest["images"]:
            x, y, z = entry["centre"]
            self.assertGreaterEqual(x, -60.0)
            self.assertLessEqual(x, 60.0)
            self.assertGreaterEqual(y, -45.0)
            self.assertLessEqual(y, 45.0)
            self.assertAlmostEqual(z, 60.0, delta=0.5)

    def test_generation_is_deterministic(self):
        second = os.path.join(self._tmp.name, "run2")
        result = subprocess.run(
            [sys.executable, SCRIPT, "--out", second],
            cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        other = os.path.join(second, "alphashape-sample")
        for name in ("manifest.json", "README.md"):
            with open(os.path.join(self.root, name), "rb") as a:
                with open(os.path.join(other, name), "rb") as b:
                    self.assertEqual(a.read(), b.read(), name + " differs between runs")
        for name in sorted(os.listdir(os.path.join(self.root, "images"))):
            with open(os.path.join(self.root, "images", name), "rb") as a:
                with open(os.path.join(other, "images", name), "rb") as b:
                    self.assertEqual(a.read(), b.read(), name + " differs between runs")


if __name__ == "__main__":
    unittest.main()
