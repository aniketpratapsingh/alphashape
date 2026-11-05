#!/usr/bin/env python3
"""WP-13.17: printable marker and chessboard sheets through Python.

Pins the two module-level helpers: a PDF starts with the PDF header, a PNG is
written by extension, and invalid code ranges or over-large boards raise
ValueError. Run with plain unittest, no pytest.
"""

import os
import tempfile
import unittest

import alphashape


class MarkerSheetTest(unittest.TestCase):
    def test_marker_sheet_pdf_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sheet.pdf")
            alphashape.make_marker_sheet(path, "circular12", count=4)
            with open(path, "rb") as handle:
                data = handle.read()
            self.assertTrue(data.startswith(b"%PDF-1.4"))
            self.assertTrue(data.endswith(b"%%EOF\n"))
            self.assertGreater(len(data), 1000)

    def test_marker_sheet_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sheet.png")
            alphashape.make_marker_sheet(path, "apriltag36h11", count=1)
            with open(path, "rb") as handle:
                signature = handle.read(8)
            self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_chessboard_sheet_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "board.pdf")
            alphashape.make_chessboard_sheet(
                path, cols=9, rows=6, square_mm=20.0
            )
            with open(path, "rb") as handle:
                self.assertTrue(handle.read(8) == b"%PDF-1.4")

    def test_rejects_code_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.pdf")
            with self.assertRaises(alphashape.ValueError):
                alphashape.make_marker_sheet(
                    path, "circular12", first_code=1000000, count=1
                )

    def test_rejects_unknown_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.pdf")
            with self.assertRaises(alphashape.ValueError):
                alphashape.make_marker_sheet(path, "nonsense")

    def test_rejects_oversized_board(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.pdf")
            with self.assertRaises(alphashape.ValueError):
                alphashape.make_chessboard_sheet(
                    path, cols=9, rows=6, square_mm=100.0
                )


if __name__ == "__main__":
    unittest.main()
