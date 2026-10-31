#!/usr/bin/env python3
"""WP-12.11: the Python CoordinateSystem surface and Chunk.crs.

What this file pins:

  * construction from an EPSG code, WKT and a PROJ string, and by built-in
    name; the name / authority / epsg / wkt properties and the repr;
  * `__eq__` is PROJ equivalence, so a WKT round trip compares equal, and the
    type is unhashable (hash / eq contract);
  * `Chunk.crs` merely relabels the chunk (no Python-side undo stack: the GUI
    owns undo and Metashape's API has none), and assigning it does NOT
    transform camera references; `CoordinateSystem()` resets to LOCAL;
  * the geoid registry lists ids and refuses a missing or non-grid file;
  * BuildDem accepts its `projection` EPSG parameter (and refuses a `crs`
    that disagrees).

When the module was built without PROJ the whole class skips: the binding is
compiled only under ALPHASHAPE_USE_PROJ, so `CoordinateSystem` does not exist.

Run with plain unittest, no pytest.
"""

import os
import tempfile
import unittest

import alphashape

HAS_CRS = hasattr(alphashape, "CoordinateSystem")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
AERIAL = [
    os.path.join(DATA_DIR, "aerial_r{}c{}.jpg".format(row, col))
    for row in range(3)
    for col in range(4)
]
UTM_ZONE_32N = 32632
UTM_ZONE_33N = 32633
WEB_MERCATOR = 3857
# 480x360 fixtures; downscale 8 keeps the DEM pipeline quick.
DOWNSCALE = 8


def _reference_repr(chunk):
    """A NaN-safe snapshot of every camera reference, for the no-transform
    assertion: two tuples holding NaN are equal as text but not as tuples."""
    return [repr(camera.reference) for camera in chunk.cameras]


@unittest.skipUnless(HAS_CRS, "module was built without PROJ")
class CoordinateSystemTest(unittest.TestCase):
    def test_from_epsg_and_name(self):
        cs = alphashape.CoordinateSystem(UTM_ZONE_33N)
        self.assertIn("UTM zone 33N", cs.name)
        self.assertEqual(cs.epsg, UTM_ZONE_33N)
        self.assertEqual(cs.authority, "EPSG")
        self.assertEqual(repr(cs), "<CoordinateSystem 'WGS 84 / UTM zone 33N'>")
        self.assertEqual(cs.wkt, cs.toWKT())
        self.assertTrue(cs.toProj4())

    def test_default_is_local(self):
        cs = alphashape.CoordinateSystem()
        self.assertIsNone(cs.epsg)
        self.assertEqual(cs.name, "LOCAL")
        self.assertEqual(repr(cs), "<CoordinateSystem 'LOCAL'>")

    def test_list_builtin(self):
        names = alphashape.CoordinateSystem.listBuiltinCRS()
        self.assertIsInstance(names, list)
        self.assertTrue(names)
        self.assertTrue(any("WGS 84" in name for name in names))
        # init() is the constructor-by-name path: "EPSG:", a built-in name
        # and WKT all resolve to the same CRS.
        by_code = alphashape.CoordinateSystem.init("EPSG::32633")
        self.assertEqual(by_code.epsg, UTM_ZONE_33N)
        self.assertEqual(by_code, alphashape.CoordinateSystem(UTM_ZONE_33N))
        self.assertEqual(alphashape.CoordinateSystem(name="EPSG:32633"), by_code)
        name = next(n for n in names if "WGS 84 / UTM zone" in n)
        cs = alphashape.CoordinateSystem.init(name)
        self.assertIsNotNone(cs.epsg)
        self.assertEqual(cs, alphashape.CoordinateSystem(cs.epsg))

    def test_wkt_roundtrip(self):
        cs = alphashape.CoordinateSystem(UTM_ZONE_33N)
        copy = alphashape.CoordinateSystem(wkt=cs.toWKT())
        self.assertEqual(copy, cs)
        self.assertEqual(copy.wkt, cs.wkt)
        self.assertEqual(copy.epsg, cs.epsg)
        self.assertEqual(alphashape.CoordinateSystem(cs), cs)

    def test_proj_roundtrip(self):
        cs = alphashape.CoordinateSystem(UTM_ZONE_33N)
        proj = cs.toProj4()
        if not proj:
            self.skipTest("toProj4 returned empty for this CRS")
        self.assertEqual(alphashape.CoordinateSystem(proj=proj), cs)

    def test_equivalence_ignores_string_identity(self):
        # The same WGS 84 geodetic CRS written lon,lat: a different WKT string
        # (axis order) that PROJ considers equivalent.
        authority = alphashape.CoordinateSystem(4326)
        lon_lat = alphashape.CoordinateSystem(
            wkt='GEOGCRS["WGS 84",'
                'DATUM["World Geodetic System 1984",'
                'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
                'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
                'CS[ellipsoidal,2],'
                'AXIS["longitude (Lon)",east,ORDER[1],'
                'ANGLEUNIT["degree",0.0174532925199433]],'
                'AXIS["latitude (Lat)",north,ORDER[2],'
                'ANGLEUNIT["degree",0.0174532925199433]]]')
        self.assertNotEqual(authority.wkt, lon_lat.wkt)
        self.assertEqual(authority, lon_lat)
        self.assertNotEqual(authority, alphashape.CoordinateSystem(UTM_ZONE_32N))
        self.assertFalse(authority == "EPSG:4326")

    def test_unhashable(self):
        # __eq__ is PROJ equivalence, not identity, so the class disables
        # identity hashing: hash(cs) must raise, as it does for tuple.
        with self.assertRaises(TypeError):
            hash(alphashape.CoordinateSystem(UTM_ZONE_33N))

    def test_invalid_inputs(self):
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(999999)
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(wkt="not a crs")
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(proj="not a proj string")
        # Exactly one of wkt= / proj= / name= is accepted: any pair is
        # ambiguous, including name= alongside a definition.
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(wkt="EPSG:32633", proj="+proj=utm +zone=33")
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(wkt="EPSG:32633", name="LOCAL")
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(proj="+proj=utm +zone=33", name="LOCAL")
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem(name="No Such CRS")
        with self.assertRaises(TypeError):
            alphashape.CoordinateSystem(bogus=1)

    def test_chunk_crs_set_relabels_only(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[AERIAL[0], AERIAL[1]])
        # A fresh chunk is the local frame.
        self.assertIsNone(chunk.crs.epsg)
        self.assertEqual(chunk.crs.name, "LOCAL")
        before = _reference_repr(chunk)
        chunk.crs = alphashape.CoordinateSystem(UTM_ZONE_33N)
        self.assertEqual(chunk.crs.epsg, UTM_ZONE_33N)
        self.assertIn("UTM zone 33N", chunk.crs.name)
        # Setting the CRS relabels the chunk; no reference is transformed.
        self.assertEqual(_reference_repr(chunk), before)
        # The GUI owns the document's undo history; there is no Python-side
        # stack (Metashape's public API has no undo either).
        self.assertFalse(hasattr(doc, "undo"))
        self.assertFalse(hasattr(doc, "undo_stack"))

    def test_chunk_crs_reset_to_local(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=[AERIAL[0]])
        chunk.crs = alphashape.CoordinateSystem(UTM_ZONE_33N)
        self.assertEqual(chunk.crs.epsg, UTM_ZONE_33N)
        before = _reference_repr(chunk)
        # An unset CoordinateSystem clears the chunk to LOCAL rather than
        # raising; the references are still untouched.
        chunk.crs = alphashape.CoordinateSystem()
        self.assertIsNone(chunk.crs.epsg)
        self.assertEqual(chunk.crs.name, "LOCAL")
        self.assertEqual(chunk.crs_epsg, 0)
        self.assertEqual(_reference_repr(chunk), before)

    def test_geoid_list_and_bad_file(self):
        ids = alphashape.CoordinateSystem.listGeoids()
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(geoid_id, str) for geoid_id in ids))
        # A missing grid is an error, never a silent no-op.
        missing = os.path.join(tempfile.gettempdir(),
                               "alphashape_definitely_missing_geoid.tif")
        if os.path.exists(missing):
            os.remove(missing)
        with self.assertRaises(alphashape.ValueError):
            alphashape.CoordinateSystem.addGeoid(missing)
        # A file that does not open as a geoid grid is refused at registration
        # (addGeoid validates the grid, and resolves the path to absolute).
        with tempfile.TemporaryDirectory() as directory:
            bogus = os.path.join(directory, "bogus_geoid.tif")
            with open(bogus, "w", encoding="utf-8") as handle:
                handle.write("this is not a GeoTIFF\n")
            with self.assertRaises(alphashape.ValueError):
                alphashape.CoordinateSystem.addGeoid(bogus)

    def test_dem_projection_param(self):
        doc = alphashape.Document()
        chunk = doc.add_chunk("Chunk 1")
        doc.active_chunk = chunk
        doc.run_task("AddPhotos", filenames=list(AERIAL))
        doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
        doc.run_task("AlignCameras")
        doc.run_task("BuildDepthMaps", downscale=DOWNSCALE, filter_mode="mild")
        # `projection` is the Int EPSG spelling the task schema declares; the
        # binder converts a Python int by the declared type. A CoordinateSystem
        # object is NOT accepted here on main (the `crs` parameter takes text).
        # Web Mercator is not the fixture's auto-UTM (32632/32633), so the
        # assertion proves `projection` is honoured rather than defaulted.
        chunk.buildDem(projection=WEB_MERCATOR, smoothing="none")
        self.assertIsNotNone(chunk.elevation)
        self.assertEqual(chunk.elevation.epsg, WEB_MERCATOR)
        # projection and crs are two spellings of one request: disagreeing is
        # refused before any work starts.
        with self.assertRaises(alphashape.ValueError):
            chunk.buildDem(projection=WEB_MERCATOR,
                           crs="EPSG:{}".format(UTM_ZONE_33N))


if __name__ == "__main__":
    unittest.main()
