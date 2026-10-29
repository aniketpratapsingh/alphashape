#!/usr/bin/env python3
"""DEM and orthomosaic views, tasks, exports, and project persistence (M9).

Milestone 9 added the BuildDem / BuildOrthomosaic tasks and their Export*
counterparts, plus the GeoTIFF side files that persist both products in the
project. What this file pins:

  * Chunk.elevation -> None, or {width, height, resolution, origin, epsg,
    nodata, source, heights() -> HxW floats, height_at(x, y) -> float|None,
    data -> raw float32 bytes}.
  * Chunk.orthomosaic -> None, or {width, height, resolution, origin, epsg,
    surface, pixels() -> HxWx4 ints, labels() -> HxW camera ids, data -> raw
    RGBA bytes}.
  * Chunk.buildDem / buildOrthomosaic / exportDem / exportOrthomosaic run the
    same tasks as doc.run_task.
  * a saved project reloads with the same grids, epsg and sampled heights.

The pipeline uses the synthetic 4x3 nadir block (data/aerial_r*c*.jpg) whose
EXIF carries exact GPS, so AlignCameras georeferences the chunk itself; the
DEM then lands in UTM zone 32N, exactly like the C++ milestone-9 tests. Depth
maps run at downscale 8 (60x45) to keep the whole file a few seconds.

Run with plain unittest, no pytest.
"""

import math
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

# 480x360 fixtures; downscale 8 keeps BuildDepthMaps and the ortho quick.
DOWNSCALE = 8
UTM_ZONE_32N = 32632


def built_document():
    """A georeferenced chunk with a DEM and an orthomosaic, built via the
    Chunk convenience methods (so the shortcut path is exercised too).

    filter_mode="mild" is pinned because this file tests the DEM/ortho VIEWS,
    not the filter: at downscale 8 on these 60x45 maps the moderate default
    (2-support) keeps 14-34% of the pixels, which leaves the DEM centre a hole
    and makes the view assertions below about emptiness rather than shape.
    """
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(AERIAL))
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
    doc.run_task("AlignCameras")
    doc.run_task("BuildDepthMaps", downscale=DOWNSCALE, filter_mode="mild")
    # interpolation="enabled" fills interior holes only: reaching 100% needs
    # the extrapolating mode, which is not what this test is about.
    chunk.buildDem(interpolation="enabled", smoothing="none")
    chunk.buildOrthomosaic(refine_seamlines=False, fill_holes=False)
    return doc, chunk


def centre_of(view):
    """World (x, y) of the middle of a gridded product."""
    ox, oy = view.origin
    rx, ry = view.resolution
    return (ox + rx * view.width / 2.0, oy - ry * view.height / 2.0)


def linear_probe(dem):
    """A few interior cells, as (col, row) and their world centre."""
    points = []
    for col, row in ((dem.width // 2, dem.height // 2),
                     (dem.width // 3, dem.height // 4),
                     (dem.width // 4, dem.height // 3)):
        ox, oy = dem.origin
        rx, ry = dem.resolution
        points.append((ox + (col + 0.5) * rx, oy - (row + 0.5) * ry))
    return points


class TestEmptyViews(unittest.TestCase):
    """Before the tasks run both products are None, not empty dicts."""

    def test_fresh_chunk_has_no_products(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(AERIAL[:1]))

        self.assertIsNone(chunk.elevation)
        self.assertIsNone(chunk.orthomosaic)

    def test_export_before_build_is_an_error(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(AERIAL[:1]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(alphashape.Error):
                chunk.exportDem(path=os.path.join(tmp, "none.tif"))
            with self.assertRaises(alphashape.Error):
                chunk.exportOrthomosaic(path=os.path.join(tmp, "none.tif"))


class TestElevationView(unittest.TestCase):
    """Chunk.elevation exposes the DEM with the declared shapes."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = built_document()
        cls.dem = cls.chunk.elevation

    def test_metadata(self):
        dem = self.dem
        self.assertIsNotNone(dem)
        self.assertGreater(dem.width, 0)
        self.assertGreater(dem.height, 0)
        self.assertEqual(dem.epsg, UTM_ZONE_32N)
        self.assertEqual(dem.source, "depth_maps")
        rx, ry = dem.resolution
        self.assertGreater(rx, 0.0)
        self.assertGreater(ry, 0.0)
        ox, oy = dem.origin
        self.assertTrue(math.isfinite(ox) and math.isfinite(oy))

    def test_heights_shape_and_bytes(self):
        dem = self.dem
        heights = dem.heights()
        self.assertEqual(len(heights), dem.height)
        for row in heights:
            self.assertEqual(len(row), dem.width)
            for value in row:
                self.assertIsInstance(value, float)
        # The raw companion is width*height float32 values, row-major.
        self.assertIsInstance(dem.data, bytes)
        self.assertEqual(len(dem.data), dem.width * dem.height * 4)

    def test_height_at_centre_is_finite(self):
        dem = self.dem
        x, y = centre_of(dem)
        h = dem.height_at(x, y)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, float)
        self.assertTrue(math.isfinite(h))
        # The scene sits ~300 m above the ellipsoid; a value near it proves the
        # UTM plane frame and the vertical path agree.
        self.assertGreater(h, 250.0)
        self.assertLess(h, 350.0)

    def test_height_at_off_grid_is_none(self):
        dem = self.dem
        ox, oy = dem.origin
        rx, ry = dem.resolution
        self.assertIsNone(dem.height_at(ox - 10.0 * rx, oy))
        self.assertIsNone(dem.height_at(ox, oy + 10.0 * ry))

    def test_nodata_cells_report_the_sentinel(self):
        dem = self.dem
        nodata = dem.nodata
        # "enabled" fill leaves the one-camera rim empty, so at least one cell
        # must carry the sentinel. height_at over a fully-empty corner is None.
        heights = dem.heights()
        seen_nodata = any(value == nodata for row in heights for value in row)
        self.assertTrue(seen_nodata)
        has_value = any(value != nodata for row in heights for value in row)
        self.assertTrue(has_value)


class TestOrthomosaicView(unittest.TestCase):
    """Chunk.orthomosaic exposes the RGBA mosaic and its seam labels."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = built_document()
        cls.ortho = cls.chunk.orthomosaic

    def test_metadata(self):
        ortho = self.ortho
        self.assertIsNotNone(ortho)
        self.assertGreater(ortho.width, 0)
        self.assertGreater(ortho.height, 0)
        self.assertEqual(ortho.epsg, UTM_ZONE_32N)
        self.assertEqual(ortho.surface, "elevation")
        # The ortho shares the DEM's plane frame.
        self.assertEqual(ortho.origin[0], self.chunk.elevation.origin[0])
        self.assertEqual(ortho.origin[1], self.chunk.elevation.origin[1])

    def test_pixels_and_bytes(self):
        ortho = self.ortho
        pixels = ortho.pixels()
        self.assertEqual(len(pixels), ortho.height)
        for row in pixels:
            self.assertEqual(len(row), ortho.width)
            for rgba in row:
                self.assertEqual(len(rgba), 4)
                for channel in rgba:
                    self.assertIsInstance(channel, int)
                    self.assertGreaterEqual(channel, 0)
                    self.assertLessEqual(channel, 255)
        self.assertIsInstance(ortho.data, bytes)
        self.assertEqual(len(ortho.data), ortho.width * ortho.height * 4)

        # At least one observed cell is opaque and at least one is nodata
        # (the fixture's outer rim is seen by no camera).
        alphas = [rgba[3] for row in pixels for rgba in row]
        self.assertIn(255, alphas)
        self.assertIn(0, alphas)

    def test_labels_reference_cameras_or_minus_one(self):
        ortho = self.ortho
        labels = ortho.labels()
        self.assertEqual(len(labels), ortho.height)
        camera_ids = {camera.id for camera in self.chunk.cameras}
        flat = [label for row in labels for label in row]
        self.assertEqual(len(flat), ortho.width * ortho.height)
        self.assertIn(-1, flat)
        for label in flat:
            if label != -1:
                self.assertIn(label, camera_ids)


class TestArgumentValidation(unittest.TestCase):
    """The convenience methods bind against the shared task schema."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = built_document()

    def test_unknown_keyword_is_an_invalid_argument(self):
        with self.assertRaises(alphashape.InvalidArgumentError):
            self.chunk.buildDem(no_such_key=1)

    def test_bad_interpolation_is_a_value_error(self):
        with self.assertRaises(alphashape.ValueError):
            self.chunk.buildDem(interpolation="bogus")

    def test_bad_blending_mode_is_a_value_error(self):
        with self.assertRaises(alphashape.ValueError):
            self.chunk.buildOrthomosaic(blending_mode="bogus")

    def test_export_without_path_is_a_value_error(self):
        with self.assertRaises(alphashape.ValueError):
            self.chunk.exportDem(path="")


class TestExport(unittest.TestCase):
    """exportDem / exportOrthomosaic write GeoTIFFs through the binder."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = built_document()

    def test_export_dem_writes_a_geotiff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dem.tif")
            self.chunk.exportDem(path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            with open(path, "rb") as handle:
                magic = handle.read(2)
            self.assertIn(magic, (b"II", b"MM"))

    def test_export_orthomosaic_writes_alpha_and_world_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ortho.tif")
            self.chunk.exportOrthomosaic(path=path, world_file=True)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "ortho.tfw")))


class TestProjectPersistence(unittest.TestCase):
    """The DEM and orthomosaic survive save/load with their grids and values."""

    def test_products_round_trip(self):
        doc, chunk = built_document()
        dem = chunk.elevation
        ortho = chunk.orthomosaic
        probes = linear_probe(dem)
        heights_before = [dem.height_at(x, y) for x, y in probes]
        pixels_before = ortho.data
        labels_before = ortho.labels()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.psx")
            doc.save(path)
            reloaded = alphashape.load_project(path)
            loaded = reloaded.chunk(reloaded.active_chunk.id)

            # The saved chunk has to be the one we loaded, not a fresh default.
            self.assertEqual(loaded.id, chunk.id)

            loaded_dem = loaded.elevation
            loaded_ortho = loaded.orthomosaic
            self.assertIsNotNone(loaded_dem)
            self.assertIsNotNone(loaded_ortho)

            self.assertEqual(loaded_dem.width, dem.width)
            self.assertEqual(loaded_dem.height, dem.height)
            self.assertEqual(loaded_dem.epsg, dem.epsg)
            self.assertEqual(loaded_dem.nodata, dem.nodata)
            self.assertEqual(loaded_dem.source, dem.source)
            self.assertEqual(loaded_dem.resolution, dem.resolution)
            self.assertEqual(loaded_dem.origin, dem.origin)
            # The labels sidecar and the raster were both written; compare the
            # whole byte image rather than a sample.
            self.assertEqual(loaded_ortho.data, pixels_before)
            self.assertEqual(loaded_ortho.labels(), labels_before)
            self.assertEqual(loaded_ortho.width, ortho.width)
            self.assertEqual(loaded_ortho.height, ortho.height)
            self.assertEqual(loaded_ortho.epsg, ortho.epsg)
            self.assertEqual(loaded_ortho.surface, ortho.surface)

            for (x, y), before in zip(probes, heights_before):
                after = loaded_dem.height_at(x, y)
                if before is None:
                    self.assertIsNone(after)
                else:
                    self.assertAlmostEqual(after, before, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
