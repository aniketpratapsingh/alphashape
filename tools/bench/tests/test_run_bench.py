#!/usr/bin/env python3
"""Unit tests for the bench runner: manifest parsing, verification and plumbing.

The pipeline itself needs the engine and real photographs, so it is not tested
here. What is tested is everything around it, because that is where a silent
mistake is most expensive: a manifest whose hash is wrong silently scores nothing,
a cache that accepts a truncated archive silently scores a partial dataset, and an
extraction that picks the wrong member silently compares against the wrong file.

Run with plain unittest (no pytest):

    python3 -m unittest discover -s tools/bench/tests -v
"""

import hashlib
import io
import json
import os
import ssl
import struct
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_bench  # noqa: E402

from contextlib import redirect_stdout  # noqa: E402


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


class ManifestTest(unittest.TestCase):
    def test_the_repository_manifest_loads_and_is_version_1(self):
        manifest = run_bench.load_manifest()
        self.assertEqual(manifest["version"], 1)
        self.assertIn("aerial-nadir", manifest["datasets"])
        self.assertIn("terrestrial-laser", manifest["datasets"])

    def test_every_dataset_has_a_licence_with_a_spdx_and_a_url(self):
        for name, dataset in run_bench.load_manifest()["datasets"].items():
            licence = dataset["licence"]
            self.assertTrue(licence.get("spdx"), name)
            self.assertTrue(licence.get("url"), name)
            self.assertTrue(licence["url"].startswith("https://"), name)

    def test_every_download_has_a_url_a_hash_and_a_byte_count(self):
        for name, dataset in run_bench.load_manifest()["datasets"].items():
            for cache_name, remote in run_bench.downloads_for(dataset):
                self.assertTrue(remote.url.startswith("https://"), (name, cache_name))
                self.assertRegex(remote.sha256, r"^[0-9a-f]{64}$", (name, cache_name))
                self.assertGreater(remote.bytes, 0, (name, cache_name))
                self.assertEqual(cache_name, Path(cache_name).name, (name, cache_name))

    def test_a_reference_without_a_url_is_served_by_another_download(self):
        # Every reference either downloads its own archive or names one that does.
        for name, dataset in run_bench.load_manifest()["datasets"].items():
            downloaded = {cache for cache, _ in run_bench.downloads_for(dataset)}
            for reference in dataset["reference_files"]:
                if reference.get("url"):
                    self.assertIn(reference["download"], downloaded, (name, reference["path"]))
                else:
                    self.assertIn(reference["from_archive"], downloaded, (name, reference["path"]))

    def test_the_images_archive_is_always_downloaded(self):
        for name, dataset in run_bench.load_manifest()["datasets"].items():
            cache_names = [cache for cache, _ in run_bench.downloads_for(dataset)]
            self.assertIn(dataset["images_archive"]["download"], cache_names, name)

    def test_describe_names_the_licence_and_every_download(self):
        dataset = run_bench.load_manifest()["datasets"]["aerial-nadir"]
        text = run_bench.describe(dataset)
        self.assertIn("CC-BY-4.0", text)
        self.assertIn("sha256", text)
        self.assertIn("zenodo.org", text)

    def test_the_cache_key_follows_the_manifest_content(self):
        manifest = run_bench.load_manifest()
        baseline = run_bench.manifest_hash(manifest)
        self.assertRegex(baseline, r"^[0-9a-f]{16}$")

        altered = json.loads(json.dumps(manifest))
        altered["datasets"]["aerial-nadir"]["images_archive"]["sha256"] = "0" * 64
        self.assertNotEqual(run_bench.manifest_hash(altered), baseline)

        reordered = json.loads(json.dumps(manifest))
        reordered["datasets"] = dict(reversed(list(reordered["datasets"].items())))
        self.assertEqual(run_bench.manifest_hash(reordered), baseline)

    def test_an_unknown_version_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as handle:
            json.dump({"version": 99, "datasets": {}}, handle)
            path = handle.name
        try:
            with self.assertRaises(SystemExit):
                run_bench.load_manifest(Path(path))
        finally:
            os.unlink(path)


class SubsetTest(unittest.TestCase):
    def _images(self, tmp: Path, count: int) -> list[Path]:
        paths = []
        for index in range(count):
            path = tmp / f"IMG_{index:04d}.JPG"
            path.write_bytes(b"x")
            paths.append(path)
        return paths

    def test_no_limit_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 10)
            selected, info = run_bench.select_subset(images, None)
            self.assertEqual(len(selected), 10)
            self.assertEqual(info["mode"], "all")

    def test_a_limit_below_the_count_selects_a_middle_block(self):
        # These files carry no readable EXIF, so the file-order fallback applies.
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 10)
            selected, info = run_bench.select_subset(images, 4)
            self.assertEqual(len(selected), 4)
            self.assertIn("file-order", info["mode"])
            # Contiguous: the names differ by the step they were taken in.
            indices = [images.index(path) for path in selected]
            self.assertEqual(indices, list(range(indices[0], indices[0] + 4)))

    def test_a_limit_above_the_count_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 5)
            selected, _ = run_bench.select_subset(images, 50)
            self.assertEqual(len(selected), 5)

    def test_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 12)
            first, _ = run_bench.select_subset(images, 5)
            second, _ = run_bench.select_subset(images, 5)
            self.assertEqual(first, second)

    def test_the_selection_names_every_image_it_chose(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 12)
            selected, info = run_bench.select_subset(images, 5)
            self.assertEqual(info["images"], [path.name for path in selected])
            self.assertEqual(info["selected"], 5)
            self.assertNotIn("first", info)  # a range would misstate a GPS cluster

    def test_a_gps_cluster_is_nearest_the_flight_centre(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = self._images(Path(tmp), 9)
            # One row of positions; the middle three are nearest the median.
            positions = [(float(index), 0.0) for index in range(9)]
            original = run_bench._image_positions
            run_bench._image_positions = lambda _images: positions
            try:
                selected, info = run_bench.select_subset(images, 3)
            finally:
                run_bench._image_positions = original
        self.assertEqual(info["mode"], "nearest-to-centroid cluster")
        self.assertEqual(
            [images.index(path) for path in selected], [3, 4, 5]
        )

    def test_dms_conversion(self):
        self.assertAlmostEqual(run_bench._dms((11.0, 40.0, 7.5477)), 11.0 + 40.0 / 60 + 7.5477 / 3600)


class MedianTest(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(run_bench.np_median([3.0, 1.0, 2.0]), 2.0)

    def test_even_count(self):
        self.assertEqual(run_bench.np_median([4.0, 1.0, 3.0, 2.0]), 2.5)

    def test_single_value(self):
        self.assertEqual(run_bench.np_median([7.0]), 7.0)


class VerifyTest(unittest.TestCase):
    def _remote(self, payload: bytes) -> run_bench.RemoteFile:
        return run_bench.RemoteFile(
            "https://example.invalid/file", hashlib.sha256(payload).hexdigest(), len(payload)
        )

    def test_a_matching_file_is_reused_without_downloading(self):
        payload = b"hello bench"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            destination.write_bytes(payload)
            # No network: the URL is unreachable, so this only works if the cached
            # file short-circuits the download.
            with redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination)
            self.assertEqual(destination.read_bytes(), payload)

    def test_a_stale_file_is_removed_before_downloading(self):
        payload = b"the real payload"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            destination.write_bytes(b"the wrong payload")
            remote = self._remote(payload)
            # Pretend the download fails, so the point is that the stale file is
            # gone rather than that the new one arrived.
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                run_bench.download(remote, destination, attempts=1, backoff_s=0)
            self.assertFalse(destination.exists())

    def test_a_hash_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            remote = run_bench.RemoteFile("https://example.invalid/x", "0" * 64, 3)
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                run_bench.download(remote, destination, attempts=1, backoff_s=0)

    def test_the_hash_of_a_known_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"abc")
            self.assertEqual(
                run_bench.sha256_file(path),
                hashlib.sha256(b"abc").hexdigest(),
            )


class DownloadRetryTest(unittest.TestCase):
    """The retry loop around ``_fetch_once``.

    The failures these cover are the ones three CI runs actually hit: a dropped TLS
    stream (``ssl.SSLEOFError`` wrapped in ``URLError``) on a big archive. The loop
    is driven directly with a stub fetch, so no network is involved and the backoff
    is zero.
    """

    def _remote(self, payload: bytes, url: str = "https://example.invalid/file") -> run_bench.RemoteFile:
        return run_bench.RemoteFile(url, hashlib.sha256(payload).hexdigest(), len(payload))

    def _with_fetch(self, fetch):
        original = run_bench._fetch_once
        run_bench._fetch_once = fetch
        self.addCleanup(setattr, run_bench, "_fetch_once", original)

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        payload = b"a payload that arrives on the second try"
        calls = []

        def fetch(remote, tmp, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError(
                    ssl.SSLEOFError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
                )
            tmp.write_bytes(payload)

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            with redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination, attempts=4, backoff_s=0)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(len(calls), 2)

    def test_a_non_transient_http_error_is_not_retried(self):
        payload = b"x"
        calls = []

        def fetch(remote, tmp, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError(remote.url, 403, "Forbidden", {}, None)

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination, attempts=4, backoff_s=0)
            self.assertEqual(len(calls), 1)

    def test_a_server_error_is_retried_until_the_attempts_run_out(self):
        payload = b"y"
        calls = []

        def fetch(remote, tmp, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError(remote.url, 503, "Service Unavailable", {}, None)

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            with self.assertRaises(SystemExit) as caught, redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination, attempts=3, backoff_s=0)
            self.assertEqual(len(calls), 3)
            self.assertIn("3 attempts", str(caught.exception))
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".zip.part").exists())

    def test_a_short_body_resumes_instead_of_restarting(self):
        payload = b"0123456789"
        seen = []

        def fetch(remote, tmp, **kwargs):
            # Emulate a transfer that ends early, then a resume that completes it.
            have = tmp.stat().st_size if tmp.exists() else 0
            seen.append(have)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "ab") as out:
                out.write(payload[have:])

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            # First attempt writes only half; the loop must keep the bytes.
            calls = {"n": 0}
            original = fetch

            def half_then_full(remote, tmp, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    tmp.write_bytes(payload[:5])
                    return
                original(remote, tmp, **kwargs)

            run_bench._fetch_once = half_then_full
            try:
                with redirect_stdout(io.StringIO()):
                    run_bench.download(self._remote(payload), destination, attempts=4, backoff_s=0)
            finally:
                run_bench._fetch_once = original
            self.assertEqual(destination.read_bytes(), payload)
            # The second attempt saw the five bytes the first one left behind.
            self.assertEqual(seen, [5])

    def test_an_unexpected_partial_with_a_bad_hash_is_discarded_before_retrying(self):
        payload = b"the good payload"
        calls = []

        def fetch(remote, tmp, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                tmp.write_bytes(b"the bad payload!")  # right length, wrong bytes
                return
            tmp.write_bytes(payload)

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            with redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination, attempts=4, backoff_s=0)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(len(calls), 2)

    def test_a_range_not_satisfiable_response_restarts_from_zero(self):
        payload = b"restart me"
        calls = []

        def fetch(remote, tmp, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                # A stale .part whose length equals the whole file.
                tmp.write_bytes(b"x" * len(payload))
                raise urllib.error.HTTPError(remote.url, 416, "Range Not Satisfiable", {}, None)
            tmp.write_bytes(payload)

        self._with_fetch(fetch)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "file.zip"
            with redirect_stdout(io.StringIO()):
                run_bench.download(self._remote(payload), destination, attempts=4, backoff_s=0)
            self.assertEqual(destination.read_bytes(), payload)


class ExtractTest(unittest.TestCase):
    def test_a_zip_is_extracted_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "images.zip"
            write_zip(archive, {"a.JPG": b"one", "nested/b.JPG": b"two"})
            out = Path(tmp) / "out"
            run_bench.extract_archive(archive, out)
            self.assertTrue((out / "a.JPG").exists())
            self.assertTrue((out / "nested" / "b.JPG").exists())

    def test_named_members_limit_the_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ref.zip"
            write_zip(
                archive,
                {"keep/ortho.tif": b"T", "keep/dem.tif": b"D", "skip/other.tif": b"X"},
            )
            out = Path(tmp) / "out"
            run_bench.extract_archive(archive, out, members=["keep/ortho.tif"])
            self.assertTrue((out / "keep" / "ortho.tif").exists())
            self.assertFalse((out / "keep" / "dem.tif").exists())
            self.assertFalse((out / "skip" / "other.tif").exists())

    def test_the_extension_decides_the_unpacker(self):
        # A zip named .7z would be a manifest error, and must fail rather than be
        # unpacked by guesswork: the extension is the only signal a download has.
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "thing.7z"
            archive.write_bytes(b"not really a 7z")
            if not any(shutil_which(name) for name in ("7z", "7za", "7zz")):
                with self.assertRaises(SystemExit):
                    run_bench.extract_archive(archive, Path(tmp) / "out")

    def test_an_unknown_archive_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "thing.rar"
            archive.write_bytes(b"whatever")
            with self.assertRaises(SystemExit):
                run_bench.extract_archive(archive, Path(tmp) / "out")


class ArchiveMemberSafetyTest(unittest.TestCase):
    """An archive is third-party input: a member that lands outside the root is
    either malformed or an attempt to write there, and both must fail the run
    with the offending name instead of silently unpacking elsewhere."""

    ESCAPES = [
        "../escape.txt",
        "../../escape.txt",
        "nested/../../escape.txt",
        "/etc/escape.txt",
        "//host/share/escape.txt",
        "C:/escape.txt",
        "c:escape.txt",
        "back\\slash.txt",
        "..",
    ]

    def test_an_escaping_name_is_rejected_by_the_checker(self):
        archive = Path("thing.zip")
        for name in self.ESCAPES:
            with self.subTest(name=name), self.assertRaises(SystemExit) as caught:
                run_bench.check_member_name(name, archive=archive)
            # The message has to name the member: "extraction failed" alone gives
            # an operator nothing to act on.
            self.assertIn(repr(name), str(caught.exception))

    def test_an_ordinary_nested_name_is_accepted(self):
        archive = Path("thing.zip")
        for name in ("a.JPG", "nested/b.JPG", "terrains/images/dslr_images/x.JPG", "."):
            with self.subTest(name=name):
                self.assertEqual(run_bench.check_member_name(name, archive=archive), name)

    def test_a_zip_carrying_a_traversal_member_is_refused_before_unpacking(self):
        # The archive itself is well formed; only the member name is hostile.
        # Refusing means nothing at all is written, including the innocent member.
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hostile.zip"
            write_zip(archive, {"ok.JPG": b"fine", "../escape.txt": b"pwned"})
            out = Path(tmp) / "out"
            with self.assertRaises(SystemExit):
                run_bench.extract_archive(archive, out)
            self.assertFalse((Path(tmp) / "escape.txt").exists(), "wrote outside the root")
            self.assertFalse((out / "ok.JPG").exists(), "extracted before checking every name")

    def test_a_zip_named_member_outside_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ref.zip"
            write_zip(archive, {"keep/ortho.tif": b"T"})
            with self.assertRaises(SystemExit):
                run_bench.extract_archive(
                    archive, Path(tmp) / "out", members=["../keep/ortho.tif"]
                )

    def test_a_tar_carrying_a_traversal_member_is_refused(self):
        import tarfile

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hostile.tar"
            payload = b"pwned"
            with tarfile.open(archive, "w") as bundle:
                info = tarfile.TarInfo("../escape.txt")
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            with self.assertRaises(SystemExit):
                run_bench.extract_archive(archive, Path(tmp) / "out")
            self.assertFalse((Path(tmp) / "escape.txt").exists())

    def test_a_7z_carrying_a_traversal_member_is_refused_before_7z_runs(self):
        # 7-Zip normalises the paths it is handed, so a hostile member cannot be
        # authored on demand; what matters is our own behaviour when one is
        # present, which is checked against the listing 7-Zip would print. The
        # binary must not be invoked at all: it honours absolute paths, so a name
        # check that ran after `x` would already have written the file.
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "hostile.7z"
            archive.write_bytes(b"stand-in: the listing is stubbed")
            out = Path(tmp) / "out"
            with mock.patch.object(
                run_bench, "seven_zip_members",
                return_value=["ok.txt", "../escape.txt"],
            ), mock.patch.object(run_bench.subprocess, "run") as ran:
                with self.assertRaises(SystemExit) as caught:
                    run_bench.extract_archive(archive, out)
            self.assertIn("escape.txt", str(caught.exception))
            ran.assert_not_called()

    def test_a_7z_named_member_that_is_absent_fails_loudly(self):
        # 7-Zip exits 0 and unpacks nothing for a name it does not hold. A
        # reference file would then be silently missing, so this has to be an
        # error naming what the archive actually contains.
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ref.7z"
            archive.write_bytes(b"stand-in")
            with mock.patch.object(
                run_bench, "seven_zip_members",
                return_value=["terrains/dslr_scan_eval/scan1.ply"],
            ):
                with self.assertRaises(SystemExit) as caught:
                    run_bench.extract_archive(
                        archive, Path(tmp) / "out",
                        members=["terrains/dslr_scan_eval/scan9.ply"],
                    )
            message = str(caught.exception)
            self.assertIn("scan9.ply", message)
            self.assertIn("scan1.ply", message)


def shutil_which(name: str):
    import shutil

    return shutil.which(name)


class CollectImagesTest(unittest.TestCase):
    def test_the_glob_and_the_declared_count_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "images").mkdir()
            for index in range(3):
                (work / "images" / f"IMG_{index}.JPG").write_bytes(b"x")
            dataset = {"title": "t", "images_archive": {"extract_glob": "*.JPG"},
                       "expected_images": 3}
            found = run_bench.collect_images(dataset, work)
            self.assertEqual([p.name for p in found], ["IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG"])

    def test_a_count_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "images").mkdir()
            (work / "images" / "only.JPG").write_bytes(b"x")
            dataset = {"title": "t", "images_archive": {"extract_glob": "*.JPG"},
                       "expected_images": 5}
            with self.assertRaises(SystemExit):
                run_bench.collect_images(dataset, work)

    def test_no_images_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "images").mkdir()
            dataset = {"title": "t", "images_archive": {"extract_glob": "*.JPG"}}
            with self.assertRaises(SystemExit):
                run_bench.collect_images(dataset, work)

    def test_without_a_glob_every_image_extension_is_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "images" / "deep").mkdir(parents=True)
            (work / "images" / "deep" / "a.JPG").write_bytes(b"x")
            (work / "images" / "b.png").write_bytes(b"x")
            (work / "images" / "notes.txt").write_text("x")
            dataset = {"title": "t", "images_archive": {}}
            found = run_bench.collect_images(dataset, work)
            self.assertEqual(len(found), 2)


class ReferencePathsTest(unittest.TestCase):
    def test_a_kind_appearing_twice_becomes_a_list(self):
        dataset = {
            "reference_files": [
                {"path": "reference/a.ply", "kind": "laser_point_cloud"},
                {"path": "reference/b.ply", "kind": "laser_point_cloud"},
                {"path": "reference/o.tif", "kind": "orthomosaic"},
            ]
        }
        paths = run_bench.reference_paths(dataset, Path("/work"))
        self.assertIsInstance(paths["laser_point_cloud"], list)
        self.assertEqual(len(paths["laser_point_cloud"]), 2)
        self.assertIsInstance(paths["orthomosaic"], str)


class ProjectInspectionTest(unittest.TestCase):
    @staticmethod
    def _project(tmp: Path, doc_xml: str) -> Path:
        project = tmp / "site.psx"
        project.write_text("alphashape-project 2\nfiles=site.files/project.zip\n")
        bundle = tmp / "site.files" / "project.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("doc.xml", doc_xml)
        return project

    def test_camera_counts_come_from_doc_xml(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <document version="2">
          <chunks>
            <chunk id="1" label="Chunk 1">
              <sensors><sensor id="0" label="s" width="100" height="100">
                <calibration type="frame" width="100" height="100" f="50" cx="0" cy="0"
                  b1="0" b2="0" k1="0" k2="0" k3="0" k4="0" p1="0" p2="0"/>
              </sensor></sensors>
              <cameras>
                <camera id="0" sensor_id="0" enabled="true" transform="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
                <camera id="1" sensor_id="0" enabled="false" transform="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
              </cameras>
            </chunk>
          </chunks>
        </document>"""
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp), xml)
            stats = run_bench.inspect_project(project)
        self.assertEqual(stats["cameras_total"], 2)
        self.assertEqual(stats["cameras_aligned"], 1)
        self.assertAlmostEqual(stats["cameras_aligned_fraction"], 0.5)
        self.assertEqual(stats["sensors"], 1)

    def test_a_missing_bundle_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = run_bench.inspect_project(Path(tmp) / "nothing.psx")
        self.assertIn("error", stats)

    def test_malformed_xml_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp), "<document><chunks>")
            stats = run_bench.inspect_project(project)
        self.assertIn("error", stats)


class EngineReportTest(unittest.TestCase):
    def test_the_alignment_line_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cli.log"
            log.write_text(
                "[info] AlignCameras: registered 30/30 cameras, RMS 0.412 px\n"
                "[info] AlignCameras: georeferenced from 30 references\n"
            )
            report = run_bench.engine_alignment_report(log)
        self.assertEqual(report["registered"], 30)
        self.assertEqual(report["offered"], 30)
        self.assertAlmostEqual(report["rms_px"], 0.412)
        self.assertTrue(report["georeferenced"])

    def test_a_log_without_the_line_gives_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cli.log"
            log.write_text("[match] 50%\n")
            self.assertEqual(run_bench.engine_alignment_report(log), {})

    def test_a_missing_log_gives_nothing(self):
        self.assertEqual(run_bench.engine_alignment_report(Path("/nonexistent/cli.log")), {})


class ReprojectionTest(unittest.TestCase):
    """The tie-point reprojection metric, end to end through the project archive.

    This path had no test and shipped a NameError that only a full CI pipeline run
    found, after the whole pipeline had succeeded. It is exercised here on a
    synthetic project whose answer is exact: a point straight down each camera's
    optical axis must reproject to its principal point.
    """

    def _project(
        self, tmp: Path, *, point=(0.0, 0.0, 10.0), camera_x=0.0, camera_type="frame"
    ) -> Path:
        project = tmp / "site.psx"
        project.write_text("alphashape-project 2\nfiles=site.files/project.zip\n")
        bundle = tmp / "site.files" / "project.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <document version="2">
          <chunks>
            <chunk id="1" label="c">
              <sensors><sensor id="0" label="s" width="100" height="100">
                <calibration type="{camera_type}" width="100" height="100" f="50" cx="0" cy="0"
                  b1="0" b2="0" k1="0" k2="0" k3="0" k4="0" p1="0" p2="0"/>
              </sensor></sensors>
              <cameras>
                <camera id="0" sensor_id="0" enabled="true"
                  transform="1 0 0 {-camera_x} 0 1 0 0 0 0 1 0 0 0 0 1"/>
              </cameras>
              <tie_points><tracks count="1"/></tie_points>
            </chunk>
          </chunks>
        </document>"""
        tracks = b"ATR2" + struct.pack("<I", 1)
        tracks += struct.pack("<I", 1) + struct.pack("<iI", 0, 0)
        tracks += b"\x01" + struct.pack("<ddd", *point)
        # Stored keypoints are pixel-CORNER based: the centre of pixel (49, 49) is
        # (49.5, 49.5) in that convention, which is (50.0, 50.0) once the +0.5
        # bridge to the calibration's pixel-centre convention is applied.
        keypoints = b"AKP1" + struct.pack("<I", 1) + struct.pack("<ffff", 49.5, 49.5, 1.0, 0.0)
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("doc.xml", xml)
            archive.writestr("chunks/1/tie_points/tracks.bin", tracks)
            archive.writestr("chunks/1/tie_points/camera_0.kpt", keypoints)
        return project

    def test_a_point_on_the_optical_axis_reprojects_to_the_principal_point(self):
        # The camera sits at x = +10 looking along +Z in its own frame, so a point
        # at (10, 0, 10) is 10 units straight in front of it: pixel (50, 50).
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp), point=(10.0, 0.0, 10.0), camera_x=10.0)
            stats = run_bench._reprojection(Path(tmp), project)
        self.assertNotIn("skipped", stats)
        self.assertEqual(stats["pairs"], 1)
        self.assertLess(stats["rms_px"], 1e-6)
        self.assertEqual(stats["tracks"], 1)

    def test_the_metric_reports_a_mismatch_as_an_error(self):
        # The keypoint's centre is (50, 50); this camera puts the point at (70, 50),
        # so the error is 20 px — the +0.5 convention bridge is applied to the
        # observed side only, and a correct bridge leaves the difference intact.
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp), point=(4.0, 0.0, 10.0), camera_x=0.0)
            stats = run_bench._reprojection(Path(tmp), project)
        self.assertAlmostEqual(stats["rms_px"], 20.0, places=6)

    def test_a_lower_case_frame_project_is_not_treated_as_fisheye(self):
        # The writer emits type="frame". When the projection compared against
        # "Frame" every frame camera took the fisheye branch: at 76 degrees off
        # axis the two models differ by 134 px on this fixture. A pinhole point at
        # x/z = 4 projects to u = 50 + 50*4 = 250, so the error from the stored
        # keypoint centre (50, 50) is exactly 200 px.
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp), point=(40.0, 0.0, 10.0), camera_x=0.0)
            stats = run_bench._reprojection(Path(tmp), project)
        self.assertNotIn("skipped", stats)
        self.assertAlmostEqual(stats["rms_px"], 200.0, places=6)

    def test_a_project_without_tracks_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "site.psx"
            project.write_text("x")
            stats = run_bench._reprojection(Path(tmp), project)
        self.assertIn("skipped", stats)


class SidecarDecodeTest(unittest.TestCase):
    def test_tracks_round_trip(self):
        payload = b"ATR2" + struct.pack("<I", 2)
        # Track 1: 2 observations, a point.
        payload += struct.pack("<I", 2)
        payload += struct.pack("<iI", 0, 4) + struct.pack("<iI", 1, 7)
        payload += b"\x01" + struct.pack("<ddd", 1.0, 2.0, 3.0)
        # Track 2: 2 observations, no point.
        payload += struct.pack("<I", 2)
        payload += struct.pack("<iI", 0, 5) + struct.pack("<iI", 2, 9)
        payload += b"\x00"

        tracks = run_bench._decode_tracks(payload)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["observations"], [(0, 4), (1, 7)])
        self.assertEqual(tracks[0]["point"], (1.0, 2.0, 3.0))
        self.assertIsNone(tracks[1]["point"])

    def test_a_bad_magic_is_rejected(self):
        with self.assertRaises(ValueError):
            run_bench._decode_tracks(b"XXXX" + struct.pack("<I", 0))

    def test_keypoints_round_trip(self):
        payload = b"AKP1" + struct.pack("<I", 2)
        payload += struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        payload += struct.pack("<ffff", 5.0, 6.0, 7.0, 8.0)
        rows = run_bench._decode_keypoints(payload)
        self.assertEqual(rows.shape, (2, 4))
        self.assertEqual(rows[1].tolist(), [5.0, 6.0, 7.0, 8.0])

    def test_cameras_and_sensors_are_decoded_from_doc_xml(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <document version="2">
          <chunks>
            <chunk id="1" label="c">
              <sensors><sensor id="3" label="s" width="10" height="10">
                <calibration type="frame" width="10" height="10" f="7" cx="1" cy="2"
                  b1="0" b2="0" k1="0.1" k2="0" k3="0" k4="0" p1="0" p2="0"/>
              </sensor></sensors>
              <cameras>
                <camera id="5" sensor_id="3" enabled="true"
                  transform="1 0 0 9 0 1 0 8 0 0 1 7 0 0 0 1"/>
              </cameras>
            </chunk>
          </chunks>
        </document>"""
        cameras = run_bench._decode_cameras(xml)
        sensors = run_bench._decode_sensors(xml)
        self.assertEqual(cameras[5]["sensor_id"], 3)
        self.assertTrue(cameras[5]["enabled"])
        self.assertEqual(cameras[5]["pose"][0, 3], 9.0)
        self.assertEqual(sensors[3]["f"], 7.0)
        self.assertEqual(sensors[3]["cx"], 1.0)
        self.assertEqual(sensors[3]["k1"], 0.1)
        # The writer's own spelling, pass-through and case-normalised.
        self.assertEqual(sensors[3]["type"], "frame")


class CommandLineTest(unittest.TestCase):
    def test_list_datasets_prints_one_line_per_dataset(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = run_bench.main(["--list-datasets"])
        self.assertEqual(status, 0)
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(len(line.split("\t")), 3)

    def test_manifest_hash_is_printed(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = run_bench.main(["--manifest-hash"])
        self.assertEqual(status, 0)
        self.assertRegex(buffer.getvalue().strip(), r"^[0-9a-f]{16}$")

    def test_list_shows_the_licence_and_the_urls(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = run_bench.main(["--dataset", "aerial-nadir", "--list"])
        self.assertEqual(status, 0)
        text = buffer.getvalue()
        self.assertIn("CC-BY-4.0", text)
        self.assertIn("https://zenodo.org", text)
        self.assertIn("MB", text)

    def test_an_unknown_dataset_is_a_usage_error(self):
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_bench.main(["--dataset", "no-such-dataset", "--list"])


class CliValueTest(unittest.TestCase):
    def test_a_bool_becomes_lowercase(self):
        # The CLI's parser accepts only lowercase true/false for a Bool parameter.
        self.assertEqual(run_bench.cli_value(True), "true")
        self.assertEqual(run_bench.cli_value(False), "false")

    def test_a_number_is_rendered_plainly(self):
        self.assertEqual(run_bench.cli_value(8), "8")
        self.assertEqual(run_bench.cli_value(0.05), "0.05")

    def test_a_string_passes_through(self):
        self.assertEqual(run_bench.cli_value("source"), "source")


class DepthParamsTest(unittest.TestCase):
    """The dispatch string is untrusted text; it is validated before it is an argv."""

    def test_empty_is_no_parameters(self):
        self.assertEqual(run_bench.parse_depth_params(""), {})
        self.assertEqual(run_bench.parse_depth_params(None), {})
        self.assertEqual(run_bench.parse_depth_params("   "), {})

    def test_a_single_pair(self):
        self.assertEqual(run_bench.parse_depth_params("cost_type=census"), {"cost_type": "census"})

    def test_several_pairs_are_split_on_whitespace(self):
        self.assertEqual(
            run_bench.parse_depth_params("  cost_type=census   max_cost=0.4\nmax_neighbors=8 "),
            {"cost_type": "census", "max_cost": "0.4", "max_neighbors": "8"},
        )

    def test_a_negative_and_a_decimal_value_are_admitted(self):
        parsed = run_bench.parse_depth_params("max_cost=-1 downscale=8 filter_mode=disabled")
        self.assertEqual(parsed["max_cost"], "-1")
        self.assertEqual(parsed["downscale"], "8")
        self.assertEqual(parsed["filter_mode"], "disabled")

    def test_a_token_without_an_equals_is_rejected(self):
        with self.assertRaises(run_bench.DepthParamError):
            run_bench.parse_depth_params("cost_type")

    def test_an_empty_key_is_rejected(self):
        with self.assertRaises(run_bench.DepthParamError):
            run_bench.parse_depth_params("=census")

    def test_an_empty_value_is_rejected(self):
        with self.assertRaises(run_bench.DepthParamError):
            run_bench.parse_depth_params("cost_type=")

    def test_an_uppercase_or_dashed_key_is_rejected(self):
        for token in ("CostType=census", "cost-type=census", "cost$type=census"):
            with self.assertRaises(run_bench.DepthParamError, msg=token):
                run_bench.parse_depth_params(token)

    def test_a_value_with_shell_or_json_metacharacters_is_rejected(self):
        # These have no legitimate use and are exactly what would be dangerous if
        # the string were ever handed to a shell.
        for token in ("cost_type=cen;sus", 'cost_type="census"', "cost_type=$(id)",
                      "cost_type=census|cat", "cost_type=census&x"):
            with self.assertRaises(run_bench.DepthParamError, msg=token):
                run_bench.parse_depth_params(token)

    def test_a_repeated_key_is_rejected(self):
        # Not "last wins": a repeated key is more likely a typo than a deliberate
        # override, and silently choosing one would make the record a lie.
        with self.assertRaises(run_bench.DepthParamError):
            run_bench.parse_depth_params("max_cost=0.4 max_cost=0.5")

    def test_the_command_line_exits_on_a_bad_string(self):
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_bench.main(["--dataset", "aerial-nadir", "--list", "--depth-params", "oops"])

    def test_the_command_line_accepts_a_good_string(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = run_bench.main(
                ["--dataset", "aerial-nadir", "--list", "--depth-params", "cost_type=census"]
            )
        self.assertEqual(status, 0)
        self.assertIn("CC-BY-4.0", out.getvalue())


class DepthParamMergeTest(unittest.TestCase):
    """The dispatch parameters are merged over the dataset's manifest overrides."""

    def dataset(self):
        return run_bench.load_manifest()["datasets"]["aerial-nadir"]

    def test_a_parameter_is_added_to_build_depth_maps(self):
        merged = run_bench.merge_depth_params(self.dataset(), {"cost_type": "census"})
        self.assertEqual(merged["param_overrides"]["BuildDepthMaps"]["cost_type"], "census")

    def test_the_dataset_s_own_overrides_survive_the_merge(self):
        # The experiment has to stay comparable with the control, so the dataset's
        # measured downscale must not be lost just because a new key was given.
        merged = run_bench.merge_depth_params(self.dataset(), {"cost_type": "census"})
        self.assertEqual(merged["param_overrides"]["BuildDepthMaps"]["downscale"], 8)
        self.assertEqual(merged["param_overrides"]["MatchPhotos"]["keypoint_limit_per_mpx"], 250)

    def test_a_parameter_overrides_the_manifest_value_of_the_same_name(self):
        merged = run_bench.merge_depth_params(self.dataset(), {"downscale": "4"})
        self.assertEqual(merged["param_overrides"]["BuildDepthMaps"]["downscale"], "4")

    def test_no_parameters_returns_the_dataset_unchanged(self):
        original = self.dataset()
        self.assertEqual(run_bench.merge_depth_params(original, {}), original)

    def test_the_input_dataset_is_not_mutated(self):
        original = self.dataset()
        before = json.dumps(original, sort_keys=True)
        run_bench.merge_depth_params(original, {"cost_type": "census"})
        self.assertEqual(json.dumps(original, sort_keys=True), before)

    def test_the_manifest_is_not_mutated(self):
        before = json.dumps(run_bench.load_manifest(), sort_keys=True)
        run_bench.merge_depth_params(self.dataset(), {"cost_type": "census"})
        self.assertEqual(json.dumps(run_bench.load_manifest(), sort_keys=True), before)

    def test_a_dataset_with_no_depth_overrides_still_gets_a_task_entry(self):
        merged = run_bench.merge_depth_params({"param_overrides": {}}, {"cost_type": "census"})
        self.assertEqual(merged["param_overrides"], {"BuildDepthMaps": {"cost_type": "census"}})


class IsExperimentTest(unittest.TestCase):
    """The workflow's one definition of "is this a control run"."""

    def status(self, payload):
        import tempfile

        import is_experiment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(payload, encoding="utf-8")
            return is_experiment.main([str(path)])

    def test_a_run_without_dispatch_parameters_is_a_control(self):
        self.assertEqual(self.status(json.dumps({"dataset": "x"})), 0)

    def test_an_empty_depth_params_dict_is_a_control(self):
        self.assertEqual(self.status(json.dumps({"depth_params": {}})), 0)

    def test_a_run_with_dispatch_parameters_is_an_experiment(self):
        self.assertEqual(self.status(json.dumps({"depth_params": {"cost_type": "census"}})), 1)

    def test_a_missing_file_cannot_tell(self):
        import is_experiment

        self.assertEqual(is_experiment.main(["/nonexistent/metrics.json"]), 2)

    def test_malformed_json_cannot_tell(self):
        self.assertEqual(self.status("{not json"), 2)

    def test_a_json_array_cannot_tell(self):
        self.assertEqual(self.status("[]"), 2)

    def test_no_argument_cannot_tell(self):
        import is_experiment

        self.assertEqual(is_experiment.main([]), 2)


class PipelineShapeTest(unittest.TestCase):
    def test_the_pipeline_order_matches_the_documented_one(self):
        # A missing or reordered stage is a silent score change, so the order the
        # README documents is pinned here.
        self.assertEqual(
            run_bench.PIPELINE,
            ["MatchPhotos", "AlignCameras", "BuildDepthMaps", "BuildPointCloud",
             "BuildDem", "BuildOrthomosaic"],
        )

    def test_every_export_names_a_task_and_a_file(self):
        for task, filename in run_bench.EXPORTS:
            self.assertTrue(task.startswith("Export"), task)
            self.assertTrue(filename.endswith((".tif", ".ply")), filename)

    def test_a_missing_cli_is_a_clear_error(self):
        original = os.environ.pop("ALPHASHAPE_CLI", None)
        try:
            with self.assertRaises(SystemExit):
                run_bench.run_cli(["--list-tasks"])
        finally:
            if original is not None:
                os.environ["ALPHASHAPE_CLI"] = original


if __name__ == "__main__":
    unittest.main()
