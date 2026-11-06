#!/usr/bin/env python3
"""Reference import/export through the Python front end (WP-13.2, WP-13.20).

What this file pins:

  * Chunk.importReference creates markers that carry a measured reference from
    a label,x,y,z CSV;
  * `marker_type` separates control points from held-out check points and
    Chunk.ground_control reports the split;
  * `columns` (import) and `columns`/`columns_order` (export) select and order
    the fields, and an unknown column name is refused;
  * an export imported back reproduces every reference.

The chunk is local (no CRS), so coordinates are read and written verbatim and
no CRS engine is required. Plain unittest, no pytest.
"""

import os
import tempfile
import unittest

import alphashape


def local_chunk():
    doc = alphashape.Document()
    chunk = doc.add_chunk("Chunk 1")
    doc.active_chunk = chunk
    return doc, chunk


def write_rows(path, rows, header=None):
    with open(path, "w", encoding="utf-8") as handle:
        if header is not None:
            handle.write(header + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


class TestImportReference(unittest.TestCase):
    def test_creates_markers_with_references(self):
        _, chunk = local_chunk()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gcps.csv")
            write_rows(path, [("GCP1", 1.0, 2.0, 3.0),
                              ("GCP2", -4.0, 5.0, 6.5)],
                       header="label,x,y,z")
            chunk.importReference(path=path)
        self.assertEqual([m.label for m in chunk.markers], ["GCP1", "GCP2"])
        self.assertEqual(chunk.markers[0].reference, (1.0, 2.0, 3.0))
        control = chunk.ground_control["control_points"]
        self.assertEqual([m.label for m in control], ["GCP1", "GCP2"])
        self.assertEqual(chunk.ground_control["check_points"], [])

    def test_marker_type_selects_check_points(self):
        _, chunk = local_chunk()
        with tempfile.TemporaryDirectory() as directory:
            control_path = os.path.join(directory, "control.csv")
            check_path = os.path.join(directory, "check.csv")
            write_rows(control_path,
                       [("C{}".format(i), 10.0 * i, 2.0, 0.5)
                        for i in range(3)], header="label,x,y,z")
            write_rows(check_path, [("CHK", 99.0, 98.0, 1.0)],
                       header="label,x,y,z")
            chunk.importReference(path=control_path, marker_type="gcp")
            chunk.importReference(path=check_path, marker_type="check_point")
        self.assertEqual(len(chunk.ground_control["control_points"]), 3)
        self.assertEqual([m.label for m in chunk.ground_control["check_points"]],
                         ["CHK"])
        self.assertEqual(len(chunk.ground_control["markers"]), 4)

    def test_explicit_columns_read_a_headerless_file(self):
        _, chunk = local_chunk()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "plain.txt")
            # Explicit columns are positional: no header row is expected.
            write_rows(path, [("P1", 7.5, -8.25, 9.0)])
            chunk.importReference(path=path, format="txt",
                                  delimiter=" ",
                                  columns=["label", "x", "y", "z"])
        self.assertEqual(chunk.markers[0].label, "P1")
        self.assertEqual(chunk.markers[0].reference, (7.5, -8.25, 9.0))

    def test_unknown_column_is_rejected(self):
        _, chunk = local_chunk()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            write_rows(path, [("P1", 1.0, 2.0, 3.0)])
            with self.assertRaises(alphashape.ValueError) as context:
                chunk.importReference(path=path, columns=["label", "banana"])
        self.assertIn("banana", str(context.exception))


class TestExportReference(unittest.TestCase):
    def test_columns_select_and_round_trip(self):
        _, source = local_chunk()
        source.addMarker(label="GCP1", position=(1.0, 2.0, 3.0))
        source.markers[0].reference = (10.0, 20.0, 30.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "out.csv")
            source.exportReference(path=path, columns=["label", "x", "y", "z"],
                                   include_cameras=False)
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines[0], "label,x,y,z")
            fields = lines[1].split(",")
            self.assertEqual(fields[0], "GCP1")
            self.assertAlmostEqual(float(fields[1]), 10.0, places=6)
            self.assertAlmostEqual(float(fields[3]), 30.0, places=6)

            _, target = local_chunk()
            target.importReference(path=path)
            self.assertEqual(target.markers[0].reference, (10.0, 20.0, 30.0))

    def test_unknown_column_is_rejected(self):
        _, chunk = local_chunk()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "out.csv")
            with self.assertRaises(alphashape.ValueError):
                chunk.exportReference(path=path, columns_order=["label", "banana"])


if __name__ == "__main__":
    unittest.main()
