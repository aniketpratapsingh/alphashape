#!/usr/bin/env python3
"""Tests for the macOS PROJ data subset selector.

    python3 packaging/tests/test_select_proj_data.py

Standard library only: this runs in the CI `checks` job, which has no test runner
installed beyond the interpreter itself.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "packaging" / "macos" / "select-proj-data.py"

MIB = 1024 * 1024


def load_selector():
    spec = importlib.util.spec_from_file_location("select_proj_data", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load_selector()


def write(path: Path, size: int, fill: bytes = b"\0") -> None:
    path.write_bytes(fill * size)


def build_source(root: Path, *, egm96_gtx: bool = False) -> Path:
    """A proj-data-shaped directory: metadata, a TIN grid and a few raster grids."""
    source = root / "share" / "proj"
    source.mkdir(parents=True)
    write(source / "proj.db", 4096)
    write(source / "proj.ini", 64)
    write(source / "README.txt", 128)
    write(source / "unrelated.md", 32)
    # PROJJSON TIN transformation: a datum shift, always shipped.
    write(source / "no_kv_ETRS89NO_NGO48_TIN.json", 2 * MIB)
    # The essentials. The EGM96 spelling is platform-dependent.
    if egm96_gtx:
        write(source / "egm96_15.gtx", 256 * 1024)
    else:
        write(source / "us_nga_egm96_15.tif", 256 * 1024)
    write(source / "us_noaa_conus.tif", 128 * 1024)
    # Regional grids competing for the budget: one small, one medium, one over the cap.
    write(source / "small_region.tif", 1 * MIB)
    write(source / "medium_region.tif", 4 * MIB)
    write(source / "huge_region.tif", 40 * MIB)
    return source


class SelectProjDataTest(unittest.TestCase):
    def test_keeps_database_tin_and_essentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_source(root)
            dest = root / "out"
            summary = selector.select(source, dest, budget_mib=16, per_file_mib=16)

            self.assertTrue((dest / "proj.db").is_file())
            self.assertTrue((dest / "proj.ini").is_file())
            self.assertTrue((dest / "README.txt").is_file())
            self.assertTrue((dest / "no_kv_ETRS89NO_NGO48_TIN.json").is_file())
            self.assertTrue((dest / "us_nga_egm96_15.tif").is_file())
            self.assertTrue((dest / "us_noaa_conus.tif").is_file())
            self.assertEqual(summary["grids"], 4)
            # Extensions PROJ does not read are not picked up by accident.
            self.assertFalse((dest / "unrelated.md").exists())

    def test_budget_is_respected_and_large_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_source(root)
            dest = root / "out"
            # 3 MB of grid budget above the ~0.4 MB of essentials: small (1 MB) fits,
            # medium (4 MB) does not, huge (40 MB) exceeds the per-file cap.
            summary = selector.select(source, dest, budget_mib=3, per_file_mib=8)

            self.assertTrue((dest / "small_region.tif").is_file())
            self.assertFalse((dest / "medium_region.tif").exists())
            self.assertFalse((dest / "huge_region.tif").exists())
            self.assertEqual(summary["dropped_budget"], 1)
            self.assertEqual(summary["dropped_large"], 1)
            grids = [n for n in summary["names"] if n.endswith((".tif", ".gtx"))]
            grid_bytes = sum((dest / n).stat().st_size for n in grids)
            self.assertLessEqual(grid_bytes, 3 * MIB + 400 * 1024)

    def test_gtx_is_accepted_as_the_egm96_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_source(root, egm96_gtx=True)
            dest = root / "out"
            summary = selector.select(source, dest)

            self.assertTrue((dest / "egm96_15.gtx").is_file())
            self.assertFalse((dest / "us_nga_egm96_15.tif").exists())
            self.assertIn("egm96_15.gtx", summary["names"])

    def test_manifest_lists_exactly_what_was_copied_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_source(root)
            first = root / "first"
            second = root / "second"
            selector.select(source, first, budget_mib=3, per_file_mib=8)
            selector.select(source, second, budget_mib=3, per_file_mib=8)

            manifest = first / selector.MANIFEST_NAME
            self.assertTrue(manifest.is_file())
            listed = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(listed, sorted(listed))
            self.assertIn("proj.db", listed)
            self.assertIn("no_kv_ETRS89NO_NGO48_TIN.json", listed)
            self.assertNotIn("medium_region.tif", listed)
            # Every listed name exists, and nothing else is on disk.
            self.assertEqual(set(listed), {p.name for p in first.iterdir()} - {selector.MANIFEST_NAME})
            # Two runs pick the same files, in the same order.
            self.assertEqual(manifest.read_bytes(),
                             (second / selector.MANIFEST_NAME).read_bytes())

    def test_missing_proj_db_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "empty"
            source.mkdir()
            with self.assertRaises(SystemExit):
                selector.select(source, root / "out")

    def test_cli_round_trip(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = build_source(root)
            dest = root / "out"
            result = subprocess.run(
                [sys.executable, str(SELECTOR), str(source), str(dest),
                 "--grid-budget-mib", "3", "--grid-max-mib", "8"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PROJ data:", result.stdout)
            self.assertTrue((dest / selector.MANIFEST_NAME).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
