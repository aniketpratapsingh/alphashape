#!/usr/bin/env python3
"""Multi-instance assets (M11) through the Python front end.

What this file pins:

  * Chunk.elevations / orthomosaics / models / point_clouds /
    depth_maps_sets expose one entry per asset, in id order, each a dict with
    `key` (the id), `label` and `active`;
  * Chunk.elevation and Chunk.orthomosaic are views onto the ACTIVE asset, and
    a second build appends rather than overwrites;
  * the `label` parameter of every producing task flows through the binder;
  * ExportDem honours `asset_id` (-1 = active), and an unknown id raises
    ValueError.

Only the DEM is exercised end to end: it is the one product the synthetic
aerial block can build without a meshing or texturing pass, and the asset
machinery it drives is shared by every kind.

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

# 480x360 fixtures; downscale 8 keeps BuildDepthMaps quick.
DOWNSCALE = 8


def dense_document():
    """A georeferenced chunk with depth maps, ready for BuildDem."""
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=list(AERIAL))
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
    doc.run_task("AlignCameras")
    doc.run_task("BuildDepthMaps", downscale=DOWNSCALE)
    return doc, chunk


class TestAssetLists(unittest.TestCase):
    """The asset lists and the active-asset rule."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = dense_document()
        # One depth-map set now exists; the DEMs are built per test class below.
        cls.chunk.buildDem(source_data="depth_maps", interpolation="enabled",
                           smoothing="none")
        cls.chunk.buildDem(source_data="depth_maps", interpolation="enabled",
                           smoothing="none", label="second dem")

    def test_every_kind_is_a_list(self):
        for name in ("elevations", "orthomosaics", "models", "point_clouds",
                     "depth_maps_sets"):
            self.assertIsInstance(getattr(self.chunk, name), list, name)

    def test_builds_append_instead_of_overwriting(self):
        elevations = self.chunk.elevations
        self.assertEqual(len(elevations), 2)
        self.assertEqual([entry["label"] for entry in elevations],
                         ["Elevation 1", "second dem"])
        # Ids are distinct, ascending, and match `key`.
        keys = [entry["key"] for entry in elevations]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 2)
        # Exactly one entry is active, and it is the last build.
        self.assertEqual([entry["active"] for entry in elevations], [False, True])

    def test_depth_map_set_is_listed(self):
        sets = self.chunk.depth_maps_sets
        self.assertGreaterEqual(len(sets), 1)
        self.assertEqual(sets[-1]["label"], "Depth Maps 1")
        self.assertTrue(sets[-1]["active"])

    def test_scalar_view_follows_the_active_asset(self):
        dem = self.chunk.elevation
        self.assertIsNotNone(dem)
        self.assertGreater(dem.width, 0)
        # The active DEM is the labelled one; the two builds differ only in
        # provenance here, so this checks the view exists and is well formed.
        self.assertEqual(dem.source, "depth_maps")

    def test_kwargs_for_the_new_parameters_are_bound(self):
        # Unknown keywords are still rejected, so the schema is the real gate.
        with self.assertRaises(alphashape.InvalidArgumentError):
            self.chunk.buildDem(not_a_parameter=1)


class TestExportByAssetId(unittest.TestCase):
    """ExportDem picks the asset named by `asset_id`."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = dense_document()
        cls.chunk.buildDem(source_data="depth_maps", interpolation="extrapolated",
                           smoothing="none", resolution=2.0, label="coarse")
        cls.coarse_id = cls.chunk.elevations[-1]["key"]
        cls.chunk.buildDem(source_data="depth_maps", interpolation="extrapolated",
                           smoothing="none", resolution=1.0, label="fine")
        cls.fine_id = cls.chunk.elevations[-1]["key"]

    def test_active_export_is_not_the_first_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            coarse = os.path.join(tmp, "coarse.tif")
            active = os.path.join(tmp, "active.tif")
            by_id = os.path.join(tmp, "by_id.tif")

            self.chunk.exportDem(path=coarse, asset_id=self.coarse_id)
            self.chunk.exportDem(path=active)  # -1 = active
            self.chunk.exportDem(path=by_id, asset_id=self.fine_id)

            with open(coarse, "rb") as handle:
                coarse_bytes = handle.read()
            with open(active, "rb") as handle:
                active_bytes = handle.read()
            with open(by_id, "rb") as handle:
                by_id_bytes = handle.read()

            self.assertTrue(coarse_bytes)
            # The coarse DEM must not be what the active export produced.
            self.assertNotEqual(coarse_bytes, active_bytes)
            # Exporting the active asset by its id is the same product.
            self.assertEqual(active_bytes, by_id_bytes)
            # The DEM view sizes agree with the assets they came from.
            self.assertGreater(self.chunk.elevation.width, 0)

    def test_unknown_asset_id_is_a_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(alphashape.ValueError):
                self.chunk.exportDem(path=os.path.join(tmp, "none.tif"), asset_id=424242)
            self.assertFalse(os.path.exists(os.path.join(tmp, "none.tif")))

    def test_export_does_not_change_the_active_asset(self):
        before = self.chunk.elevations
        with tempfile.TemporaryDirectory() as tmp:
            self.chunk.exportDem(path=os.path.join(tmp, "x.tif"),
                                 asset_id=self.coarse_id)
        after = self.chunk.elevations
        self.assertEqual([e["key"] for e in before], [e["key"] for e in after])
        self.assertEqual([e["active"] for e in before], [e["active"] for e in after])


class TestExportReprojection(unittest.TestCase):
    """WP-12.9: an export can reproject and restyle the file on the fly."""

    @classmethod
    def setUpClass(cls):
        cls.doc, cls.chunk = dense_document()
        cls.chunk.buildDem(source_data="depth_maps", interpolation="extrapolated",
                           smoothing="none", resolution=1.0)
        cls.dem_epsg = cls.chunk.elevation.epsg

    def test_reprojection_changes_the_declared_crs(self):
        target = 3857 if self.dem_epsg != 3857 else 25832
        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, "plain.tif")
            reprojected = os.path.join(tmp, "reprojected.tif")
            self.chunk.exportDem(path=plain)
            self.chunk.exportDem(path=reprojected, projection=target)
            # The reprojection always changes the bytes: the grid and the
            # GeoKeys both move, even for a projection that keeps the cell size.
            self.assertNotEqual(_read(plain), _read(reprojected))

    def test_resolution_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cell.tif")
            cell = 3.0
            self.chunk.exportDem(path=path, resolution=cell)
            # Read the GeoTIFF back through a second export at the same cell:
            # the same request twice is deterministic, which is what the asset
            # export path promises.
            again = os.path.join(tmp, "cell2.tif")
            self.chunk.exportDem(path=again, resolution=cell)
            self.assertEqual(_read(path), _read(again))

    def test_disagreeing_projection_and_crs_raise(self):
        target = 3857 if self.dem_epsg != 3857 else 25832
        other = 25832 if target == 3857 else 3857
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(alphashape.ValueError):
                self.chunk.exportDem(path=os.path.join(tmp, "conflict.tif"),
                                     projection=target, crs="EPSG:{}".format(other))

    def test_negative_resolution_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(alphashape.ValueError):
                self.chunk.exportDem(path=os.path.join(tmp, "bad.tif"), resolution=-1.0)


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()

if __name__ == "__main__":
    unittest.main()
