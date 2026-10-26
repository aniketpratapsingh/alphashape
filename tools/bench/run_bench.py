#!/usr/bin/env python3
"""Download, run and measure the alphashape CLI on a public dataset.

One invocation drives a whole benchmark dataset:

  1. read ``datasets.json`` and lay out a cache directory,
  2. download each archive from its publisher and verify its SHA-256 (byte count
     first, so a truncated download fails fast instead of hashing a partial file),
  3. extract the images and the reference products,
  4. run the CLI pipeline task by task on the default parameters, timing each
     stage and sampling peak resident memory,
  5. hand the products to ``metrics.evaluate`` and write the metrics JSON.

Everything is addressable by path so the CI job can cache steps 2-3 across runs.
A run is reproducible from the manifest alone: no imagery is committed and no
step reaches a network resource that is not named in ``datasets.json``.

Usage:

    python3 tools/bench/run_bench.py --dataset aerial-nadir --work build/bench
    python3 tools/bench/run_bench.py --dataset aerial-nadir --list
    python3 tools/bench/run_bench.py --virtualenv build/bench/venv ...

``--list`` prints what the dataset would download and from where, without
touching the network. It is the check to run after editing the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets.json"

# The bench is a set of scripts, not an installed package: ``metrics`` is a
# sibling module, and putting its directory on the path is what lets the file run
# both as ``python3 tools/bench/run_bench.py`` and under the test discovery.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402 - after the path setup above

# The pipeline, in the order the README documents it. Defaults only: the point of
# a benchmark is to measure what a user gets without tuning, so a run that needs a
# parameter tweak to work is a finding, not something to work around here.
PIPELINE = [
    "MatchPhotos",
    "AlignCameras",
    "BuildDepthMaps",
    "BuildPointCloud",
    "BuildDem",
    "BuildOrthomosaic",
]

# Products exported at the end, by task and destination.
EXPORTS = [
    ("ExportDem", "dem.tif"),
    ("ExportOrthomosaic", "ortho.tif"),
    ("ExportPointCloud", "cloud.ply"),
]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class RemoteFile:
    """One download: a URL, the expected SHA-256 and the expected byte count."""

    url: str
    sha256: str
    bytes: int


def load_manifest(path: Path = MANIFEST) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise SystemExit(f"{path}: unsupported manifest version {manifest.get('version')!r}")
    return manifest


def manifest_hash(manifest: dict) -> str:
    """A stable digest of the manifest, for the workflow cache key.

    Cache invalidation has to follow the *content* of the manifest, not the file's
    mtime: a renamed dataset or a corrected hash must produce a different key so
    the cache is not reused against a dataset that no longer matches.
    """
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def downloads_for(dataset: dict) -> list[tuple[str, RemoteFile]]:
    """Every (cache file name, remote file) pair a dataset needs, deduplicated.

    ``download`` is the name the file takes in the cache, and its extension is what
    decides how it is unpacked. It is deliberately independent of the extracted
    path: a reference unzips to ``reference/ortho.tif`` but arrives as
    ``processed.zip``, and naming the cache file after the extracted path would make
    the archive look like a GeoTIFF.

    Several reference members can come out of one archive; that archive appears
    once here and the caller extracts each member from it, rather than fetching it
    again per member.
    """
    out: list[tuple[str, RemoteFile]] = []
    seen: set[str] = set()

    images = dataset["images_archive"]
    out.append(
        (
            images["download"],
            RemoteFile(images["url"], images["sha256"], int(images["bytes"])),
        )
    )
    seen.add(images["url"])

    for reference in dataset["reference_files"]:
        url = reference.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            (
                reference["download"],
                RemoteFile(url, reference["sha256"], int(reference["bytes"])),
            )
        )
    return out


def describe(dataset: dict) -> str:
    """Human-readable summary for ``--list`` and the workflow's job summary."""
    lines = [
        f"{dataset['title']}",
        f"  publisher: {dataset['publisher']}",
        f"  licence:   {dataset['licence']['spdx']} ({dataset['licence']['url']})",
        f"  flight:    {dataset['flight']}",
    ]
    if dataset.get("doi"):
        lines.append(f"  doi:       {dataset['doi']}")
    total = 0
    for name, remote in downloads_for(dataset):
        total += remote.bytes
        lines.append(f"  download:  {name} <- {remote.url}")
        lines.append(f"             {remote.bytes:,} bytes, sha256 {remote.sha256}")
    lines.append(f"  total download: {total / 1e6:.1f} MB")
    references = [ref["kind"] for ref in dataset["reference_files"]]
    lines.append(f"  reference products: {', '.join(references) if references else 'none'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download and verify
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(remote: RemoteFile, destination: Path) -> None:
    """Fetch ``remote`` to ``destination``, verifying size and hash.

    A file already present with the right hash is left alone, which is what makes
    a cached work directory cheap. A file with the right size but the wrong hash
    is deleted rather than overwritten in place, so a subsequent run re-downloads
    instead of hashing a corrupt cache every time.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == remote.bytes and sha256_file(destination) == remote.sha256:
            print(f"  cached   {destination.name} ({remote.bytes:,} bytes)")
            return
        print(f"  stale    {destination.name}: removing")
        destination.unlink()

    print(f"  download {remote.url}")
    print(f"           -> {destination} ({remote.bytes:,} bytes)")
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        remote.url, headers={"User-Agent": "alphashape-quality-bench/1.0"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    elapsed = time.monotonic() - started

    size = tmp.stat().st_size
    if size != remote.bytes:
        tmp.unlink()
        raise SystemExit(
            f"{remote.url}: expected {remote.bytes:,} bytes, got {size:,}"
        )
    actual = sha256_file(tmp)
    if actual != remote.sha256:
        tmp.unlink()
        raise SystemExit(
            f"{remote.url}: sha256 mismatch\n  expected {remote.sha256}\n  actual   {actual}"
        )
    tmp.rename(destination)
    rate = size / elapsed / 1e6 if elapsed > 0 else 0.0
    print(f"  verified {destination.name} in {elapsed:.1f} s ({rate:.1f} MB/s)")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_archive(archive: Path, destination: Path, *, members: list[str] | None = None) -> None:
    """Extract ``archive`` under ``destination``.

    ``members`` limits the extraction to the named entries, which is how a single
    reference archive is unpacked once even when several reference files come out
    of it. Zip and tar (including .tar.zst handled by the external `tar`) are
    supported; a `.7z` needs `7z`, which the workflow installs.
    """
    destination.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            if members is None:
                bundle.extractall(destination)
            else:
                for member in members:
                    bundle.extract(member, destination)
        return

    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.zst", ".tar.xz", ".tar.bz2")):
        with tarfile.open(archive) as bundle:
            if members is None:
                bundle.extractall(destination)
            else:
                for member in members:
                    bundle.extract(member, destination)
        return

    if name.endswith(".7z"):
        seven = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
        if seven is None:
            raise SystemExit(
                f"{archive.name}: extracting a .7z needs 7-Zip on PATH "
                "(the workflow installs it with `apt-get install p7zip-full`)"
            )
        command = [seven, "x", "-y", f"-o{destination}", str(archive)]
        if members:
            command.extend(members)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return

    raise SystemExit(f"{archive}: unsupported archive type")


def collect_images(dataset: dict, work: Path) -> list[Path]:
    """The image files the manifest says the archive supplies, sorted by name."""
    pattern = dataset["images_archive"].get("extract_glob")
    images_root = work / "images"
    if pattern:
        found = sorted(images_root.glob(pattern))
    else:
        found = sorted(
            path
            for path in images_root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
    expected = dataset.get("expected_images")
    if expected is not None and len(found) != expected:
        raise SystemExit(
            f"{dataset['title']}: expected {expected} images, found {len(found)} under "
            f"{images_root} with pattern {pattern!r}"
        )
    if not found:
        raise SystemExit(f"{dataset['title']}: no images matched {pattern!r}")
    return found


def reference_paths(dataset: dict, work: Path) -> dict:
    """Where each reference product landed, keyed by ``kind``.

    A kind that appears more than once (ETH3D ships two laser scans for the same
    scene) becomes a list, so the caller can score against either or both.
    """
    out: dict = {}
    for reference in dataset["reference_files"]:
        destination = str(work / reference["path"])
        kind = reference["kind"]
        if kind in out:
            existing = out[kind]
            out[kind] = existing + [destination] if isinstance(existing, list) else [existing, destination]
        else:
            out[kind] = destination
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_cli(args: list[str], *, log: Path | None = None) -> float:
    """Run one CLI invocation, returning its wall time in seconds.

    A non-zero exit is fatal: the benchmark is about a pipeline that works on
    real photographs, so a stage that errors is a finding to fix, not a number to
    record as zero. stderr goes to ``log`` when one is given, which is what makes
    a CI failure diagnosable after the fact.
    """
    cli = os.environ.get("ALPHASHAPE_CLI")
    if not cli:
        raise SystemExit("ALPHASHAPE_CLI is not set; see tools/bench/README.md")
    command = [cli, *args]
    print(f"  $ {' '.join(command)}")
    started = time.monotonic()
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as errlog:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errlog)
    else:
        result = subprocess.run(command, stdout=subprocess.DEVNULL)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise SystemExit(
            f"alphashape-cli exited {result.returncode}: {' '.join(args)}"
            + (f" (see {log})" if log else "")
        )
    return elapsed


def run_pipeline(work: Path, images: list[Path]) -> dict:
    """Create a project from ``images`` and run the pipeline, timing each stage.

    Peak RSS is taken from ``/proc/self/status`` on Linux or ``ru_maxrss`` from
    ``getrusage``: the CLI is a short-lived child, so the reliable figure is the
    child's own reported peak, read through ``wait4``. Where neither is available
    the field is ``None`` rather than a guess.
    """
    project = work / "site.psx"
    log = work / "cli.log"
    stages: dict[str, float] = {}

    started = time.monotonic()
    stage_log_start = log.stat().st_size if log.exists() else 0
    stages["AddPhotos"] = _timed(run_cli, ["--new", str(project), "--add-images",
                                           *[str(p) for p in images]], log=log)
    for task in PIPELINE:
        stages[task] = _timed(run_cli, ["--project", str(project), "--task", task], log=log)
    total = time.monotonic() - started

    products: dict[str, str] = {}
    for task, filename in EXPORTS:
        destination = work / "products" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        stages[task] = _timed(
            run_cli,
            ["--project", str(project), "--task", task, "--set", f"path={destination}"],
            log=log,
        )
        if not destination.exists():
            raise SystemExit(f"{task} reported success but {destination} is missing")
        products[filename] = str(destination)

    return {
        "stages_s": stages,
        "total_s": total,
        "project": str(project),
        "products": products,
        "peak_rss_mb": peak_rss_mb(),
        "log_bytes": log.stat().st_size if log.exists() else 0,
        "log_tail": _tail(log, 20),
    }


def peak_rss_mb() -> float | None:
    """Peak RSS of this process tree, in MB, or None when the platform hides it."""
    try:
        import resource  # noqa: PLC0415 - POSIX only

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        # macOS reports bytes, Linux kilobytes; both are covered by the platform
        # check rather than a magic constant.
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):  # pragma: no cover - non-POSIX
        return None


def _timed(function, *args, **kwargs):
    """Call ``function`` and return its own elapsed seconds (it reports them)."""
    return function(*args, **kwargs)


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return [line.rstrip("\n") for line in content[-lines:]]


# ---------------------------------------------------------------------------
# Project inspection (camera statistics)
# ---------------------------------------------------------------------------


def inspect_project(project: Path) -> dict:
    """Camera-count and alignment statistics from the saved project.

    Reads the .psx manifest, opens ``<stem>.files/project.zip`` and parses
    ``doc.xml`` for the sensors and cameras. Kept here rather than in ``metrics``
    because it is project-format knowledge, not metric maths; the numbers it feeds
    (aligned cameras, reprojection error) are computed by ``metrics``.

    A project that cannot be read returns ``{"error": ...}`` instead of raising:
    the run already failed loudly if a stage broke, and an unreadable project
    after a successful pipeline is a finding worth reporting, not a crash.
    """
    try:
        base = project.with_suffix("")
        bundle = base.parent / f"{base.name}.files" / "project.zip"
        if not bundle.exists():
            return {"error": f"{bundle} not found"}
        with zipfile.ZipFile(bundle) as archive:
            xml = archive.read("doc.xml").decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"error": f"cannot read project: {exc}"}

    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        return {"error": f"doc.xml is not well-formed: {exc}"}

    chunk = root.find("./chunks/chunk")
    if chunk is None:
        return {"error": "doc.xml has no chunk"}
    cameras = chunk.findall("./cameras/camera")
    aligned = [c for c in cameras if c.get("enabled") == "true"]
    sensors = chunk.findall("./sensors/sensor")
    return {
        "cameras_total": len(cameras),
        "cameras_aligned": len(aligned),
        "cameras_aligned_fraction": (len(aligned) / len(cameras)) if cameras else 0.0,
        "sensors": len(sensors),
        "has_transform": chunk.find("./transform") is not None,
        "epsg": (chunk.find("./crs").get("epsg") if chunk.find("./crs") is not None else None),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(dataset_name: str, work: Path, *, keep: bool = False) -> dict:
    manifest = load_manifest()
    if dataset_name not in manifest["datasets"]:
        raise SystemExit(
            f"unknown dataset {dataset_name!r}; known: {', '.join(sorted(manifest['datasets']))}"
        )
    dataset = manifest["datasets"][dataset_name]
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"== {dataset['title']} ({dataset_name})")
    print(f"   licence {dataset['licence']['spdx']}")

    print("-- fetch")
    for path, remote in downloads_for(dataset):
        download(remote, cache / Path(path).name)

    print("-- extract")
    archive = cache / dataset["images_archive"]["download"]
    image_root = work / "images"
    if not any(image_root.rglob("*")):
        extract_archive(archive, image_root)
    else:
        print(f"  cached   {image_root}")

    # References: one extraction per download, then the named members. Keyed by the
    # cache file name so a single archive that supplies several members (two laser
    # scans, both GeoTIFFs from one ODM run) is unpacked once.
    by_download: dict[str, list[str]] = {}
    for reference in dataset["reference_files"]:
        source = reference.get("from_archive") or reference["download"]
        by_download.setdefault(source, [])
        if reference.get("archive_member"):
            by_download[source].append(reference["archive_member"])

    for name, members in by_download.items():
        destination = work / "reference_raw" / Path(name).stem
        if destination.exists() and any(destination.rglob("*")):
            print(f"  cached   {destination}")
            continue
        extract_archive(cache / name, destination, members=members or None)

    # Promote the reference members to their manifest paths.
    reference_root = work / "reference"
    for reference in dataset["reference_files"]:
        target = work / reference["path"]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_name = reference.get("from_archive") or reference["download"]
        source_root = work / "reference_raw" / Path(archive_name).stem
        member = reference.get("archive_member")
        if member is None:
            found = next(iter(source_root.rglob("*")))
        else:
            found = source_root / member
        if not found.exists():
            raise SystemExit(f"{dataset_name}: reference {member!r} missing after extraction")
        shutil.copy2(found, target)
    references = reference_paths(dataset, work)

    print("-- pipeline")
    images = collect_images(dataset, work)
    print(f"   {len(images)} images")
    timing = run_pipeline(work, images)

    print("-- inspect")
    camera_stats = inspect_project(Path(timing["project"]))
    if "error" not in camera_stats:
        camera_stats["reprojection"] = _reprojection(work, Path(timing["project"]))
    else:
        print(f"   {camera_stats['error']}")

    print("-- metrics")
    spec = metrics.DatasetSpec(
        name=dataset_name,
        images=[str(p) for p in images],
        ortho_reference=references.get("orthomosaic"),
        dem_reference=references.get("dsm"),
        laser_reference=references.get("laser_point_cloud"),
        nominal_gsd_m=dataset.get("nominal_gsd_m"),
        licence=dataset["licence"],
    )
    products = timing["products"]
    result = metrics.evaluate(
        spec,
        ortho_path=products.get("ortho.tif"),
        dem_path=products.get("dem.tif"),
        cloud_path=products.get("cloud.ply"),
        camera_stats=camera_stats,
    )
    result["timing"] = timing
    result["manifest_hash"] = manifest_hash(manifest)
    result["dataset_manifest"] = {
        "title": dataset["title"],
        "doi": dataset.get("doi"),
        "flight": dataset["flight"],
        "nominal_gsd_m": dataset.get("nominal_gsd_m"),
        "images": len(images),
    }
    return result


def _reprojection(work: Path, project: Path) -> dict:
    """Tie-point reprojection RMS from the saved project.

    This is the one reference-free accuracy number available without ground truth:
    re-project every triangulated tie point through every observing camera's pose
    and calibration, then compare with the keypoint the detector actually found.
    It measures the alignment's internal consistency, not its absolute accuracy —
    a self-consistent but wrongly scaled model scores well here, which is why the
    report never presents it alone.

    The binaries are read straight out of the project archive so this works on a
    machine with no Python bindings built.
    """
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    base = project.with_suffix("")
    bundle = base.parent / f"{base.name}.files" / "project.zip"
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            tracks_entry = next((n for n in names if n.endswith("/tie_points/tracks.bin")), None)
            if tracks_entry is None:
                return {"skipped": "project has no tracks"}
            prefix = tracks_entry.rsplit("/tie_points/", 1)[0]
            tracks = _decode_tracks(archive.read(tracks_entry))

            keypoints: dict[int, np.ndarray] = {}
            for name in names:
                if name.startswith(prefix + "/tie_points/camera_") and name.endswith(".kpt"):
                    camera_id = int(name.rsplit("_", 1)[1].split(".")[0])
                    keypoints[camera_id] = _decode_keypoints(archive.read(name))

            cameras = _decode_cameras(archive.read("doc.xml"))
            sensors = _decode_sensors(archive.read("doc.xml"))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"skipped": f"cannot read tie points: {exc}"}

    observed = []
    projected = []
    for track in tracks:
        if track["point"] is None:
            continue
        point = np.asarray([track["point"]], dtype=np.float64)
        for camera_id, keypoint_index in track["observations"]:
            camera = cameras.get(camera_id)
            if camera is None:
                continue
            sensor = sensors.get(camera["sensor_id"])
            if sensor is None:
                continue
            pixels, ok = metrics.project_pixels(point, camera["pose"], sensor)
            if not ok[0]:
                continue
            kp = keypoints.get(camera_id)
            if kp is None or keypoint_index >= kp.shape[0]:
                continue
            observed.append(kp[keypoint_index][:2])
            projected.append(pixels[0])

    if not observed:
        return {"skipped": "no projectable observations"}
    stats = metrics.reprojection_stats(
        np.asarray(observed, dtype=np.float64), np.asarray(projected, dtype=np.float64)
    )
    stats["tracks"] = len(tracks)
    return stats


def _decode_tracks(data: bytes) -> list[dict]:
    """Decode the engine's ATR2/ATR1 tracks sidecar (see common/io/sidecars.hpp)."""
    if len(data) < 8 or data[:4] != b"ATR2":
        if len(data) < 8 or data[:4] != b"ATR1":
            raise ValueError("tracks.bin: bad magic")
    magic = data[:4]
    count = struct.unpack_from("<I", data, 4)[0]
    at = 8
    tracks = []
    for _ in range(count):
        observations = struct.unpack_from("<I", data, at)[0]
        at += 4
        observations_list = []
        for _ in range(observations):
            camera_id, keypoint_index = struct.unpack_from("<iI", data, at)
            at += 8
            observations_list.append((camera_id, keypoint_index))
        has_point = data[at]
        at += 1
        point = None
        if has_point:
            point = struct.unpack_from("<ddd", data, at)
            at += 24
        tracks.append({"observations": observations_list, "point": point})
    return tracks


def _decode_keypoints(data: bytes) -> "object":
    """Decode the engine's AKP1 keypoints sidecar into an (n, 4) float array."""
    import numpy as np  # noqa: PLC0415

    if len(data) < 8 or data[:4] != b"AKP1":
        raise ValueError("keypoints: bad magic")
    count = struct.unpack_from("<I", data, 4)[0]
    at = 8
    rows = np.empty((count, 4), dtype=np.float64)
    for index in range(count):
        rows[index] = struct.unpack_from("<ffff", data, at)
        at += 16
    return rows


def _decode_cameras(xml_bytes: bytes) -> dict:
    """Parse <cameras> out of doc.xml: world->camera pose per camera id."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    chunk = root.find("./chunks/chunk")
    out: dict = {}
    if chunk is None:
        return out
    for camera in chunk.findall("./cameras/camera"):
        transform = camera.get("transform")
        if transform is None:
            continue
        values = [float(v) for v in transform.split()]
        if len(values) != 16:
            continue
        out[int(camera.get("id"))] = {
            "sensor_id": int(camera.get("sensor_id", -1)),
            "pose": np.asarray(values, dtype=np.float64).reshape(4, 4),
            "enabled": camera.get("enabled") == "true",
        }
    return out


def _decode_sensors(xml_bytes: bytes) -> dict:
    """Parse <sensors> out of doc.xml: the Brown-Conrady calibration per sensor."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    chunk = root.find("./chunks/chunk")
    out: dict = {}
    if chunk is None:
        return out
    for sensor in chunk.findall("./sensors/sensor"):
        calibration = sensor.find("./calibration")
        if calibration is None:
            continue
        entry = {
            "type": calibration.get("type", "Frame"),
            "width": int(calibration.get("width", 0)),
            "height": int(calibration.get("height", 0)),
        }
        for field in ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"):
            entry[field] = float(calibration.get(field, 0.0))
        out[int(sensor.get("id"))] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="dataset name from datasets.json")
    parser.add_argument("--work", default="build/bench", help="work directory (default: build/bench)")
    parser.add_argument("--json", help="write the metrics JSON here")
    parser.add_argument("--list", action="store_true", help="describe the dataset and exit")
    parser.add_argument("--list-datasets", action="store_true", help="list dataset names and exit")
    parser.add_argument("--manifest-hash", action="store_true", help="print the cache key and exit")
    args = parser.parse_args(argv)

    manifest = load_manifest()

    if args.list_datasets:
        for name, dataset in sorted(manifest["datasets"].items()):
            print(f"{name}\t{dataset['licence']['spdx']}\t{dataset['title']}")
        return 0

    if args.manifest_hash:
        print(manifest_hash(manifest))
        return 0

    if not args.dataset:
        parser.error("--dataset is required")

    dataset = manifest["datasets"].get(args.dataset)
    if dataset is None:
        parser.error(f"unknown dataset {args.dataset!r}")

    if args.list:
        print(describe(dataset))
        return 0

    result = run(args.dataset, Path(args.work))

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"-- wrote {destination}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
