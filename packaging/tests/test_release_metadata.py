#!/usr/bin/env python3
"""Tests for the release metadata generators (WP-29.8).

    python3 packaging/tests/test_release_metadata.py

Standard library only: this runs in the CI `checks` job, which has no test runner
installed beyond the interpreter itself. Exits non-zero on the first failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENSES = ROOT / "packaging" / "licenses"
SBOM = ROOT / "packaging" / "sbom"

sys.path.insert(0, str(LICENSES))
sys.path.insert(0, str(SBOM))

from collect import render  # noqa: E402
from depsfile import PLATFORMS, parse_deps, platforms_for  # noqa: E402
from make_sbom import build_document, normalise_timestamp  # noqa: E402

DEPS = LICENSES / "deps.yaml"


class DepsFileTest(unittest.TestCase):
    def test_every_entry_has_the_required_fields(self):
        entries = parse_deps(DEPS)
        self.assertGreater(len(entries), 10)
        for entry in entries:
            for key in ("name", "copyright", "spdx", "url"):
                self.assertTrue(entry[key], f"{entry['name']}: empty {key}")

    def test_platforms_are_known_and_non_empty(self):
        for entry in parse_deps(DEPS):
            self.assertTrue(entry["platforms"], entry["name"])
            for platform in entry["platforms"]:
                self.assertIn(platform, PLATFORMS)

    def test_no_duplicate_components(self):
        names = [entry["name"] for entry in parse_deps(DEPS)]
        self.assertEqual(len(names), len(set(names)))

    def test_rejects_an_unknown_platform(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("- name: X\n  copyright: Y\n  spdx: MIT\n  url: z\n  platforms: [beos]\n")
            path = handle.name
        with self.assertRaises(ValueError):
            parse_deps(path)
        Path(path).unlink()

    def test_rejects_a_missing_required_field(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("- name: X\n  copyright: Y\n  url: z\n")
            path = handle.name
        with self.assertRaisesRegex(ValueError, "missing required"):
            parse_deps(path)
        Path(path).unlink()

    def test_platform_filter(self):
        entries = parse_deps(DEPS)
        for platform in PLATFORMS:
            selected = platforms_for(entries, platform)
            self.assertTrue(selected, platform)
            for entry in selected:
                self.assertIn(platform, entry["platforms"])


class NoticeTest(unittest.TestCase):
    def test_notice_lists_every_component(self):
        entries = parse_deps(DEPS)
        text = render(entries, "v9.9.9", "https://example.invalid/repo")
        for entry in entries:
            self.assertIn(entry["name"], text, f"{entry['name']} is missing from NOTICE.txt")
            self.assertIn(entry["spdx"], text)
            self.assertIn(entry["url"], text)
        self.assertIn("v9.9.9", text)
        self.assertIn(str(len(entries)), text)

    def test_notice_has_no_unrendered_placeholder(self):
        text = render(parse_deps(DEPS), "v9.9.9", "https://example.invalid/repo")
        self.assertNotIn("@", text.replace("aniketpratapsingh", ""))

    def test_notice_is_deterministic(self):
        entries = parse_deps(DEPS)
        once = render(entries, "v1", "https://example.invalid")
        twice = render(parse_deps(DEPS), "v1", "https://example.invalid")
        self.assertEqual(once, twice)

    def test_cli_writes_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "NOTICE.txt"
            result = subprocess.run(
                [sys.executable, str(LICENSES / "collect.py"),
                 "--version", "v1", "--out", str(out)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file())
            self.assertIn("THIRD-PARTY NOTICES", out.read_text(encoding="utf-8"))


class SbomTest(unittest.TestCase):
    NOW = "2026-10-23T12:00:00Z"

    def test_document_shape(self):
        entries = parse_deps(DEPS)
        doc = build_document(entries, "v0.2.0", self.NOW, "https://example.invalid/repo", "abc123")
        self.assertEqual(doc["spdxVersion"], "SPDX-2.3")
        self.assertEqual(doc["dataLicense"], "CC0-1.0")
        self.assertEqual(doc["SPDXID"], "SPDXRef-DOCUMENT")
        self.assertEqual(doc["creationInfo"]["created"], self.NOW)
        self.assertTrue(doc["documentNamespace"].startswith("https://example.invalid/repo/spdx/"))

        # One package per component plus the application itself.
        self.assertEqual(len(doc["packages"]), len(entries) + 1)
        ids = [package["SPDXID"] for package in doc["packages"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate SPDXID")

        # Every component appears, and the application names each of them.
        described = {rel["relatedSpdxElement"] for rel in doc["relationships"]
                     if rel["relationshipType"] == "DEPENDS_ON"}
        for entry in entries:
            self.assertTrue(any(package["name"] == entry["name"] for package in doc["packages"]),
                            f"{entry['name']} is missing from the SBOM")
        self.assertEqual(len(described), len(entries))

        # A DESCRIBES relationship from the document to the application.
        self.assertTrue(any(rel["relationshipType"] == "DESCRIBES"
                            and rel["spdxElementId"] == "SPDXRef-DOCUMENT"
                            for rel in doc["relationships"]))

    def test_every_package_carries_the_required_fields(self):
        doc = build_document(parse_deps(DEPS), "v1", self.NOW, "https://example.invalid", "r")
        for package in doc["packages"]:
            for key in ("name", "SPDXID", "versionInfo", "downloadLocation", "filesAnalyzed",
                        "licenseConcluded", "licenseDeclared", "copyrightText"):
                self.assertIn(key, package, f"{package.get('name')}: no {key}")
            self.assertIs(package["filesAnalyzed"], False)
            self.assertNotEqual(package["licenseConcluded"], "")

    def test_relationships_reference_existing_ids(self):
        doc = build_document(parse_deps(DEPS), "v1", self.NOW, "https://example.invalid", "r")
        known = {package["SPDXID"] for package in doc["packages"]} | {"SPDXRef-DOCUMENT"}
        for rel in doc["relationships"]:
            self.assertIn(rel["spdxElementId"], known)
            self.assertIn(rel["relatedSpdxElement"], known)

    def test_rejects_a_non_expression_licence(self):
        entries = parse_deps(DEPS)
        entries[0]["spdx"] = "see the website"
        with self.assertRaises(ValueError):
            build_document(entries, "v1", self.NOW, "https://example.invalid", "r")

    def test_creation_info_creators_are_valid_spdx_forms(self):
        # SPDX 2.3 section 6.8: "<type>: <value>", and for Organization the value is
        # "name (email)" -- a URL in the parentheses is not a valid creator string.
        doc = build_document(parse_deps(DEPS), "v1", self.NOW, "https://example.invalid", "r")
        creators = doc["creationInfo"]["creators"]
        self.assertIn("Tool: alphashape-sbom", creators)
        organisations = [c for c in creators if c.startswith("Organization: ")]
        self.assertEqual(len(organisations), 1)
        for creator in creators:
            self.assertRegex(creator, r"^(Tool|Organization|Person): \S")
            if creator.startswith("Organization: "):
                name = creator[len("Organization: "):]
                self.assertNotIn("http", name)
                if "(" in name:
                    self.assertIn("@", name.split("(", 1)[1])

    def test_rejects_a_bad_timestamp(self):
        for value in ("2026-10-23T12:00:00", "not a date", "2026-10-23T12:00:00+02:00"):
            with self.assertRaises(ValueError, msg=value):
                normalise_timestamp(value)
        self.assertEqual(normalise_timestamp(self.NOW), self.NOW)

    def test_cli_output_is_valid_json_and_byte_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.json"
            second = Path(tmp) / "b.json"
            for out in (first, second):
                result = subprocess.run(
                    [sys.executable, str(SBOM / "make_sbom.py"), "--version", "v1",
                     "--out", str(out), "--now", self.NOW, "--ref", "abc"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            document = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(document["spdxVersion"], "SPDX-2.3")
            self.assertTrue(first.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
